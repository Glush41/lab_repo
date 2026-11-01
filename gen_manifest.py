import hashlib, json, os

files = [
    "data/cifar-100-python/train",
    "data/cifar-100-python/test",
    "data/cifar-100-python/meta"
]
manifest = []
for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024*1024), b""):
            h.update(chunk)
    manifest.append({
        "path": f,
        "sha256": h.hexdigest(),
        "size_bytes": os.path.getsize(f)
    })
with open("hash_manifest.json", "w", encoding="utf-8") as out:
    json.dump(manifest, out, indent=2)
print("done")
