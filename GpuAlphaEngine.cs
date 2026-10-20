using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DifferenceCutoutPng;

/// <summary>
/// Runs the embedded difference_matte.onnx model on the GPU through DirectML.
/// TryCompute returns false on ANY problem (no DirectML-capable adapter, missing native
/// onnxruntime/DirectML DLLs, driver error, ...) so the caller can fall back to the CPU path.
/// </summary>
internal static class GpuAlphaEngine
{
    private const string ModelResource = "DifferenceCutoutPng.difference_matte.onnx";
    private const int PixelsPerBand = 2_000_000;   // keeps tensor memory bounded on huge images

    private static InferenceSession? _session;
    private static bool _unavailable;              // sticky: don't retry a GPU that already failed
    private static readonly object Gate = new();

    public static bool TryCompute(byte[] comp, byte[] bg, int w, int h, MatteSettings s, float[] alpha)
    {
        try
        {
            InferenceSession? session = GetSession();
            if (session is null) return false;

            int rowsPerBand = Math.Max(16, PixelsPerBand / Math.Max(1, w));

            var tol = new DenseTensor<float>(new[] { s.Tolerance }, new[] { 1 });
            var soft = new DenseTensor<float>(new[] { s.Softness }, new[] { 1 });
            var lw = new DenseTensor<float>(new[] { s.LightnessWeight }, new[] { 1 });

            for (int y0 = 0; y0 < h; y0 += rowsPerBand)
            {
                int rows = Math.Min(rowsPerBand, h - y0);
                var dims = new[] { 1, 3, rows, w };

                var canvasTensor = new DenseTensor<float>(ToPlanar(comp, w, y0, rows), dims);
                var bgTensor = new DenseTensor<float>(ToPlanar(bg, w, y0, rows), dims);

                using var results = session.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("canvas", canvasTensor),
                    NamedOnnxValue.CreateFromTensor("background", bgTensor),
                    NamedOnnxValue.CreateFromTensor("tolerance", tol),
                    NamedOnnxValue.CreateFromTensor("softness", soft),
                    NamedOnnxValue.CreateFromTensor("lightness_weight", lw)
                });

                float[] band = results.First().AsTensor<float>().ToArray();
                if (band.Length != rows * w) return false;
                Array.Copy(band, 0, alpha, (long)y0 * w, band.Length);
            }

            return true;
        }
        catch
        {
            // Anything at all going wrong on the GPU path -> CPU takes over.
            lock (Gate)
            {
                _unavailable = true;
                _session?.Dispose();
                _session = null;
            }
            return false;
        }
    }

    private static InferenceSession? GetSession()
    {
        lock (Gate)
        {
            if (_unavailable) return null;
            if (_session is not null) return _session;

            try
            {
                using var stream = typeof(GpuAlphaEngine).Assembly.GetManifestResourceStream(ModelResource)
                                   ?? throw new InvalidOperationException("Embedded model not found.");
                using var ms = new MemoryStream();
                stream.CopyTo(ms);

                using var options = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,   // required by the DirectML provider
                    EnableMemoryPattern = false                      // required by the DirectML provider
                };
                options.AppendExecutionProvider_DML(0);              // throws if DirectML is unavailable

                var session = new InferenceSession(ms.ToArray(), options);

                // Warm-up on a tiny image: surfaces driver/adapter problems now, not mid-save.
                var dims = new[] { 1, 3, 4, 4 };
                var zeros = new DenseTensor<float>(new float[3 * 16], dims);
                using (session.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("canvas", zeros),
                    NamedOnnxValue.CreateFromTensor("background", zeros),
                    NamedOnnxValue.CreateFromTensor("tolerance", new DenseTensor<float>(new[] { 10f }, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("softness", new DenseTensor<float>(new[] { 8f }, new[] { 1 })),
                    NamedOnnxValue.CreateFromTensor("lightness_weight", new DenseTensor<float>(new[] { 1f }, new[] { 1 }))
                })) { }

                _session = session;
                return _session;
            }
            catch
            {
                _unavailable = true;
                return null;
            }
        }
    }

    /// <summary>BGRA bytes (rows y0..y0+rows) -> planar RGB floats 0..255 in NCHW order.</summary>
    private static float[] ToPlanar(byte[] bgra, int w, int y0, int rows)
    {
        int plane = rows * w;
        var data = new float[3 * plane];

        Parallel.For(0, rows, ry =>
        {
            int src = (y0 + ry) * w * 4;
            int dst = ry * w;
            for (int x = 0; x < w; x++, src += 4, dst++)
            {
                data[dst] = bgra[src + 2];               // R
                data[plane + dst] = bgra[src + 1];       // G
                data[2 * plane + dst] = bgra[src];       // B
            }
        });

        return data;
    }
}
