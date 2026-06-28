namespace Stt.Abstractions.Models;

/// <summary>
/// Catalog of installed models (spec §5.1, §10.3). The built-in directory may be empty
/// (models are user-supplied, D7); the registry scans <c>LocalAppData/models/</c> plus user
/// import paths.
/// </summary>
public interface IModelRegistry
{
    /// <summary>All known models.</summary>
    IReadOnlyList<ModelManifest> List();

    /// <summary>Get a model by id, or throw <see cref="KeyNotFoundException"/>.</summary>
    ModelManifest Get(string id);

    /// <summary>
    /// Sideload from a folder: load <c>manifest.json</c> if present, else infer a tentative
    /// manifest from file naming + ONNX metadata for the user to confirm (spec §10.3).
    /// </summary>
    ModelManifest ImportFromFolder(string folderPath);

    /// <summary>Remove a model from the registry (does not delete files unless implementation chooses).</summary>
    void Remove(string id);
}
