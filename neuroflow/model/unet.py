#FIXME test Unet

class Unet(nn.Module):
    def __init__(self, levels = 1, kernel_shape = [[3,3,2,8],
                                                  [3,3,8,16],
                                                  [3,3,16,32],
                                                  [3,3,32,64]]):
        super(G, self).__init__()
        self.blocks = []
        for i in range(levels-1):
            self.downsamples.append(self.block(kernel_shape[i]))

        self.mid = self.block(kernel_shape[-1])
        self.final = self.block(kernel_shape[0], down=True)
        self.levels = levels
        self.upsamples = []
        for i in range(1, levels):
            self.upsamples.append(self.block(kernel_shape[-i], down=True))

        self.pool = nn.MaxPool2d(2))
        self.upsample = nn.Upsample(size=2)

    #FIXME add residuals if needed
    def block(self, shape, padding=1, down=True):
        inp, out = shape[2], shape[3]
        if down:
            inp, out = shape[3], shape[2]

        return nn.Sequential(
                nn.Conv2d(inp, out, kernel_size=shape[0], padding=pad),
                nn.ReLU(True))

    def forward(self, x):
        layers = [x]
        for i in range(self.levels-1):
            x = self.downsamples[i](x)
            layers.append(x)
            x = self.pool(x)

        x = self.mid(x)
        layers.append(x)

        for i in range(1, self.levels):
            x = self.upsample(x)
            x_enc  = layers[(self.levels-1)-(i-1)]
            x = self.upsamples[i](x+x_enc)
        x = self.final(x)
        return x
