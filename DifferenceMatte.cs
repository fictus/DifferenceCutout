using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using PaintDotNet;

namespace DifferenceCutoutPng;

internal sealed class MatteSettings
{
    /// <summary>Lab distance below which a pixel counts as "same as background" (alpha 0).</summary>
    public float Tolerance { get; init; } = 10f;

    /// <summary>Width of the soft ramp above Tolerance, in Lab units (0 = hard edge).</summary>
    public float Softness { get; init; } = 8f;

    /// <summary>1 = lightness differences count fully, 0 = ignore lightness (shadows) and compare colour only.</summary>
    public float LightnessWeight { get; init; } = 0.6f;

    /// <summary>Number of 3x3 median passes on the alpha mask (removes specks, fills pinholes).</summary>
    public int CleanupPasses { get; init; } = 1;

    /// <summary>Remove background colour bleed from semi-transparent edge pixels.</summary>
    public bool Decontaminate { get; init; } = true;

    /// <summary>Auto = GPU (DirectML) when available, otherwise CPU. Cpu = never touch the GPU.</summary>
    public ComputeDevice Device { get; init; } = ComputeDevice.Auto;
}

internal enum ComputeDevice
{
    Auto,
    Cpu
}

/// <summary>
/// Difference matting: given a base image B and a canvas image C that is B plus a foreground,
/// estimate per-pixel alpha from the perceptual (CIELAB) distance between C and B, then
/// recover the foreground colour. Fully local, deterministic, no network.
/// </summary>
internal static class DifferenceMatte
{
    private static readonly float[] SrgbToLinear = BuildLut();

    // ---------------------------------------------------------------- public API

    /// <summary>Loads the base image as straight-alpha BGRA bytes at exactly w x h.</summary>
    public static byte[] LoadBackground(string path, int w, int h, bool stretchIfDifferent)
    {
        using var src = new Bitmap(path);

        if ((src.Width != w || src.Height != h) && !stretchIfDifferent)
        {
            throw new InvalidOperationException(
                $"The background image is {src.Width}x{src.Height} but the canvas is {w}x{h}. " +
                "Use an image with the same size, or choose \"Stretch to canvas\".");
        }

        using var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(bmp))
        using (var attrs = new ImageAttributes())
        {
            attrs.SetWrapMode(WrapMode.TileFlipXY);
            g.CompositingMode = CompositingMode.SourceCopy;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
            g.DrawImage(src, new Rectangle(0, 0, w, h), 0, 0, src.Width, src.Height, GraphicsUnit.Pixel, attrs);
        }

        var data = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            var result = new byte[w * h * 4];
            for (int y = 0; y < h; y++)
            {
                IntPtr row = data.Scan0 + y * data.Stride;
                System.Runtime.InteropServices.Marshal.Copy(row, result, y * w * 4, w * 4);
            }
            return result;
        }
        finally
        {
            bmp.UnlockBits(data);
        }
    }

    /// <summary>Builds the transparent cutout. Caller owns (and must dispose) the returned surface.</summary>
    public static Surface Compute(Surface canvas, byte[] bg, MatteSettings s, Action<double>? progress = null)
    {
        int w = canvas.Width, h = canvas.Height;
        var comp = new byte[w * h * 4];   // canvas composited over the background (BGRA, opaque)
        var alpha = new float[w * h];

        progress?.Invoke(0.05);

        // 0) composite the canvas over the base image
        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int i = (y * w + x) * 4;

                ColorBgra c = canvas[x, y];
                int a = c.A;
                byte bB = bg[i], bG = bg[i + 1], bR = bg[i + 2];

                byte cB = (byte)((c.B * a + bB * (255 - a) + 127) / 255);
                byte cG = (byte)((c.G * a + bG * (255 - a) + 127) / 255);
                byte cR = (byte)((c.R * a + bR * (255 - a) + 127) / 255);

                comp[i] = cB; comp[i + 1] = cG; comp[i + 2] = cR; comp[i + 3] = 255;

            }
        });

        progress?.Invoke(0.2);

        // 1) perceptual distance -> alpha: GPU model when possible, otherwise parallel CPU
        bool onGpu = s.Device == ComputeDevice.Auto && GpuAlphaEngine.TryCompute(comp, bg, w, h, s, alpha);
        if (!onGpu)
        {
            ComputeAlphaCpu(comp, bg, w, h, s, alpha);
        }
        LastDevice = onGpu ? "GPU (DirectML)" : "CPU";

        progress?.Invoke(0.55);

        // 2) clean-up
        for (int pass = 0; pass < s.CleanupPasses; pass++)
        {
            alpha = Median3x3(alpha, w, h);
        }

        progress?.Invoke(0.8);

        // 3) write output with optional colour decontamination on soft edges
        var result = new Surface(w, h);
        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int p = y * w + x;
                int i = p * 4;
                float a = alpha[p];

                if (a <= 0.0039f)
                {
                    result[x, y] = ColorBgra.FromBgra(0, 0, 0, 0);
                    continue;
                }

                float b = comp[i], g = comp[i + 1], r = comp[i + 2];

                if (s.Decontaminate && a < 0.999f)
                {
                    float inv = 1f / a, k = 1f - a;
                    b = Clamp255((b - k * bg[i]) * inv);
                    g = Clamp255((g - k * bg[i + 1]) * inv);
                    r = Clamp255((r - k * bg[i + 2]) * inv);
                }

                result[x, y] = ColorBgra.FromBgra((byte)(b + 0.5f), (byte)(g + 0.5f), (byte)(r + 0.5f),
                                                  (byte)Math.Clamp(a * 255f + 0.5f, 0f, 255f));
            }
        });

        progress?.Invoke(1.0);
        return result;
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>Which device did the last Compute call use (for diagnostics).</summary>
    public static string LastDevice { get; private set; } = "CPU";

    private static void ComputeAlphaCpu(byte[] comp, byte[] bg, int w, int h, MatteSettings s, float[] alpha)
    {
        Parallel.For(0, h, y =>
        {
            for (int x = 0; x < w; x++)
            {
                int p = y * w + x;
                int i = p * 4;

                ToLab(comp[i + 2], comp[i + 1], comp[i], out float l1, out float a1, out float b1);
                ToLab(bg[i + 2], bg[i + 1], bg[i], out float l2, out float a2, out float b2);

                float dl = (l1 - l2) * s.LightnessWeight;
                float da = a1 - a2;
                float db = b1 - b2;
                alpha[p] = Ramp(MathF.Sqrt(dl * dl + da * da + db * db), s.Tolerance, s.Softness);
            }
        });
    }

    private static float Ramp(float d, float tolerance, float softness)
    {
        if (softness < 0.001f) return d > tolerance ? 1f : 0f;
        float t = Math.Clamp((d - tolerance) / softness, 0f, 1f);
        return t * t * (3f - 2f * t);
    }

    private static float Clamp255(float v) => v < 0f ? 0f : (v > 255f ? 255f : v);

    private static float[] Median3x3(float[] src, int w, int h)
    {
        var dst = new float[src.Length];
        Parallel.For(0, h, y =>
        {
            Span<float> win = stackalloc float[9];
            for (int x = 0; x < w; x++)
            {
                int n = 0;
                for (int dy = -1; dy <= 1; dy++)
                {
                    int yy = Math.Clamp(y + dy, 0, h - 1);
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int xx = Math.Clamp(x + dx, 0, w - 1);
                        win[n++] = src[yy * w + xx];
                    }
                }
                win.Sort();
                dst[y * w + x] = win[4];
            }
        });
        return dst;
    }

    private static float[] BuildLut()
    {
        var lut = new float[256];
        for (int i = 0; i < 256; i++)
        {
            float c = i / 255f;
            lut[i] = c <= 0.04045f ? c / 12.92f : MathF.Pow((c + 0.055f) / 1.055f, 2.4f);
        }
        return lut;
    }

    private static void ToLab(byte r, byte g, byte b, out float L, out float A, out float B)
    {
        float rl = SrgbToLinear[r], gl = SrgbToLinear[g], bl = SrgbToLinear[b];

        float x = (0.4124564f * rl + 0.3575761f * gl + 0.1804375f * bl) / 0.95047f;
        float y = 0.2126729f * rl + 0.7151522f * gl + 0.0721750f * bl;
        float z = (0.0193339f * rl + 0.1191920f * gl + 0.9503041f * bl) / 1.08883f;

        float fx = F(x), fy = F(y), fz = F(z);
        L = 116f * fy - 16f;
        A = 500f * (fx - fy);
        B = 200f * (fy - fz);
    }

    private static float F(float t) => t > 0.008856f ? MathF.Cbrt(t) : 7.787f * t + 16f / 116f;
}
