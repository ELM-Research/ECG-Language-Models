class Signal:
    def __call__(self, opened_npy):
        print("hi")

c = Signal()
v = {"l" : c}
print(v)

# v["l"]("kl")

if type(c).__name__ not in ["rgb", "s"]:
    print(type(c).__name__)
    print(type(type(c).__name__))