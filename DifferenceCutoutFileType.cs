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
/// "Save as" type that stores only what differs between the canvas and a chosen base image
/// as a transparent PNG. Save-only: it never appears in the Open dialog.
/// </summary>
public sealed class DifferenceCutoutFileType : PropertyBasedFileType
{
    public const string TypeName = "Difference Cutout PNG";

    private enum PropertyNames
    {
        BackgroundImage,
        SizeMismatch,
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

    private enum SizeMode
    {
        Error,
        Stretch
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
            new StringProperty(PropertyNames.BackgroundImage, string.Empty, 1024),
            new StaticListChoiceProperty(PropertyNames.SizeMismatch,
                new object[] { SizeMode.Error, SizeMode.Stretch }, 0),
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

        ui.SetPropertyControlValue(PropertyNames.BackgroundImage, ControlInfoPropertyNames.DisplayName,
            "Background base image (required)");
        ui.SetPropertyControlType(PropertyNames.BackgroundImage, PropertyControlType.FileChooser);

        ui.SetPropertyControlValue(PropertyNames.SizeMismatch, ControlInfoPropertyNames.DisplayName,
            "If sizes differ");
        PropertyControlInfo sizeInfo = ui.FindControlForPropertyName(PropertyNames.SizeMismatch);
        sizeInfo.SetValueDisplayName(SizeMode.Error, "Show an error");
        sizeInfo.SetValueDisplayName(SizeMode.Stretch, "Stretch to canvas");

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
        string path = token.GetProperty<StringProperty>(PropertyNames.BackgroundImage).Value?.Trim() ?? string.Empty;

        if (path.Length == 0 || !File.Exists(path))
        {
            // Paint.NET calls Save in the background to draw the options-dialog preview the moment the
            // dialog opens, before a base image can possibly have been chosen. Don't fail there: show the
            // unchanged image. A real save (OK button) without a base image still fails with a clear message.
            if (IsOptionsDialogPreview())
            {
                input.Flatten(scratchSurface);
                WritePng(scratchSurface, output);
                return;
            }

            if (path.Length == 0)
            {
                throw new InvalidOperationException(
                    "Difference Cutout PNG needs a background base image. Click Save again and choose one in the options dialog.");
            }
            throw new FileNotFoundException("The background base image could not be found.", path);
        }

        var sizeMode = (SizeMode)token.GetProperty<StaticListChoiceProperty>(PropertyNames.SizeMismatch).Value;

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

        // Flatten all layers into the scratch surface (same size as the document).
        input.Flatten(scratchSurface);
        Report(0.02);

        byte[] background = DifferenceMatte.LoadBackground(
            path, scratchSurface.Width, scratchSurface.Height, sizeMode == SizeMode.Stretch);

        using Surface cutout = DifferenceMatte.Compute(scratchSurface, background, settings, Report);

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