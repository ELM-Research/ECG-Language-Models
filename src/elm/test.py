class Signal:
    def __call__(self, opened_npy):
        print("hi")

c = Signal()
v = {"l" : c}
print(v)

v["l"]("kl")

