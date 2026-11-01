with open("data/cifar-100-python/meta", "r+b") as f:
    f.seek(100)
    b = f.read(1)
    f.seek(100)
    f.write(bytes([b[0] ^ 0xFF]))
print("tampered")
