using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using PaintDotNet;
using PaintDotNet.IndirectUI;
using PaintDotNet.PropertySystem;

namespace DifferenceCutoutPng;

public sealed class PluginSupportInfo : IPluginSupportInfo
{
    public string Author => "Your Name";
    public string Copyright => "Copyright (c) Your Name";
    public string DisplayName => DifferenceCutoutFileType.TypeName;
    public Version Version => typeof(PluginSupportInfo).Assembly.GetName().Version ?? new Version(1, 0, 0, 0);
    public Uri WebsiteUri => new Uri("https://www.getpaint.net/");
}

public sealed class DifferenceCutoutFileTypeFactory : IFileTypeFactory
{
    public FileType[] GetFileTypeInstances() => new FileType[] { new DifferenceCutoutFileType() };
}

/// <summary>
/// "Save as" type that stores only what differs between the foreground layer and a chosen background layer
/// as a transparent PNG. Save-only: it never appears in the Open dialog.
/// </summary>
public sealed class DifferenceCutoutFileType : PropertyBasedFileType
{
    public const string TypeName = "Difference Cutout PNG";

    private enum PropertyNames
    {
        BackgroundLayer,
        Device,
        Tolerance,
        Softness,
        ShadowSensitivity,
        Cleanup,
        Decontaminate
    }

    private enum DeviceChoice
    {
        Auto,
        Cpu
    }

    public DifferenceCutoutFileType()
        : base(TypeName, new FileTypeOptions
        {
            LoadExtensions = Array.Empty<string>(),
            SaveExtensions = new[] { ".png" }
        })
    {
    }

    // ------------------------------------------------------------------ load (not supported)

    // FileType declares OnLoad as abstract, so it must exist even for a save-only type.
    // LoadExtensions is empty, so Paint.NET never routes Open requests here.
    protected override Document OnLoad(Stream input)
    {
        throw new NotSupportedException($"{TypeName} is a save-only file type.");
    }

    // ------------------------------------------------------------------ save dialog

    public override PropertyCollection OnCreateSavePropertyCollection()
    {
        var props = new List<Property>
        {
            new Int32Property(PropertyNames.BackgroundLayer, 1, 1, 99),
            new StaticListChoiceProperty(PropertyNames.Device,
                new object[] { DeviceChoice.Auto, DeviceChoice.Cpu }, 0),
            new Int32Property(PropertyNames.Tolerance, 10, 0, 60),
            new Int32Property(PropertyNames.Softness, 8, 0, 40),
            new Int32Property(PropertyNames.ShadowSensitivity, 60, 0, 100),
            new Int32Property(PropertyNames.Cleanup, 1, 0, 3),
            new BooleanProperty(PropertyNames.Decontaminate, true)
        };

        return new PropertyCollection(props);
    }

    public override ControlInfo OnCreateSaveConfigUI(PropertyCollection props)
    {
        ControlInfo ui = CreateDefaultSaveConfigUI(props);

        ui.SetPropertyControlValue(PropertyNames.BackgroundLayer, ControlInfoPropertyNames.DisplayName,
            "Background layer number (1 = bottom layer)");

        ui.SetPropertyControlValue(PropertyNames.Device, ControlInfoPropertyNames.DisplayName,
            "Processing");
        PropertyControlInfo deviceInfo = ui.FindControlForPropertyName(PropertyNames.Device);
        deviceInfo.SetValueDisplayName(DeviceChoice.Auto, "Auto (GPU, falls back to CPU)");
        deviceInfo.SetValueDisplayName(DeviceChoice.Cpu, "CPU only");

        ui.SetPropertyControlValue(PropertyNames.Tolerance, ControlInfoPropertyNames.DisplayName,
            "Difference tolerance");
        ui.SetPropertyControlValue(PropertyNames.Softness, ControlInfoPropertyNames.DisplayName,
            "Edge softness");
        ui.SetPropertyControlValue(PropertyNames.ShadowSensitivity, ControlInfoPropertyNames.DisplayName,
            "Shadow sensitivity (low = ignore shadows)");
        ui.SetPropertyControlValue(PropertyNames.Cleanup, ControlInfoPropertyNames.DisplayName,
            "Speck clean-up passes");

        ui.SetPropertyControlValue(PropertyNames.Decontaminate, ControlInfoPropertyNames.DisplayName, string.Empty);
        ui.SetPropertyControlValue(PropertyNames.Decontaminate, ControlInfoPropertyNames.Description,
            "Remove background colour from soft edges");

        return ui;
    }

    // ------------------------------------------------------------------ save

    protected override void OnSaveT(
        Document input,
        Stream output,
        PropertyBasedSaveConfigToken token,
        Surface scratchSurface,
        ProgressEventHandler progressCallback)
    {
        // Paint.NET calls Save in the background to draw the options-dialog preview. Never throw there
        // (a failing preview breaks the dialog); show the unchanged image instead. A real save still
        // reports problems with a clear message.
        bool preview = IsOptionsDialogPreview();

        try
        {
            SaveCutout(input, output, token, scratchSurface, progressCallback);
        }
        catch (Exception) when (preview)
        {
            output.SetLength(0);
            input.Flatten(scratchSurface);
            WritePng(scratchSurface, output);
        }
    }

    private void SaveCutout(
        Document input,
        Stream output,
        PropertyBasedSaveConfigToken token,
        Surface scratchSurface,
        ProgressEventHandler progressCallback)
    {
        int w = input.Width, h = input.Height;

        var bitmapLayers = new List<BitmapLayer>();
        foreach (var layer in input.Layers)
        {
            if (layer is BitmapLayer bl) bitmapLayers.Add(bl);
        }

        if (bitmapLayers.Count < 2)
        {
            throw new InvalidOperationException(
                "Difference Cutout PNG needs at least two layers: a background layer and the layer to cut out.");
        }

        int bgNumber = token.GetProperty<Int32Property>(PropertyNames.BackgroundLayer).Value;
        if (bgNumber < 1 || bgNumber > bitmapLayers.Count)
        {
            string names = string.Join(", ", bitmapLayers.Select((l, i) => $"{i + 1} = \"{l.Name}\""));
            throw new InvalidOperationException(
                $"Background layer number {bgNumber} doesn't exist; the image has {bitmapLayers.Count} layers " +
                $"(numbered from the bottom: {names}). Click Save again and pick a valid layer number.");
        }

        BitmapLayer bgLayer = bitmapLayers[bgNumber - 1];
        var foreground = bitmapLayers.Where(l => !ReferenceEquals(l, bgLayer) && l.Visible).ToList();
        if (foreground.Count == 0)
        {
            throw new InvalidOperationException(
                "There is no visible layer other than the background layer to cut out.");
        }

        var device = (DeviceChoice)token.GetProperty<StaticListChoiceProperty>(PropertyNames.Device).Value;

        var settings = new MatteSettings
        {
            Device = device == DeviceChoice.Cpu ? ComputeDevice.Cpu : ComputeDevice.Auto,
            Tolerance = token.GetProperty<Int32Property>(PropertyNames.Tolerance).Value,
            Softness = token.GetProperty<Int32Property>(PropertyNames.Softness).Value,
            LightnessWeight = token.GetProperty<Int32Property>(PropertyNames.ShadowSensitivity).Value / 100f,
            CleanupPasses = token.GetProperty<Int32Property>(PropertyNames.Cleanup).Value,
            Decontaminate = token.GetProperty<BooleanProperty>(PropertyNames.Decontaminate).Value
        };

        void Report(double fraction) => progressCallback(this, new ProgressEventArgs(fraction * 100.0));

        byte[] background = DifferenceMatte.SurfaceToBytes(bgLayer.Surface);
        Report(0.02);

        using Surface canvas = DifferenceMatte.CompositeLayers(w, h, foreground);
        using Surface cutout = DifferenceMatte.Compute(canvas, background, settings, Report);

        WritePng(cutout, output);
    }

    /// <summary>True when Save was invoked by the options dialog's live preview rather than a real save.</summary>
    private static bool IsOptionsDialogPreview()
    {
        try
        {
            foreach (StackFrame frame in new StackTrace(false).GetFrames())
            {
                string? name = frame.GetMethod()?.DeclaringType?.FullName;
                if (name is not null && name.Contains("SaveConfigDialog", StringComparison.Ordinal))
                {
                    return true;
                }
            }
        }
        catch
        {
            // fall through: treat as a real save
        }
        return false;
    }

    private static void WritePng(Surface surface, Stream output)
    {
        // System.Drawing's PNG encoder wants a seekable stream, so encode to memory first.
        using var bitmap = surface.CreateAliasedBitmap();
        using var buffer = new MemoryStream();
        bitmap.Save(buffer, ImageFormat.Png);
        buffer.Position = 0;
        buffer.CopyTo(output);
    }
}