require 'nn'
require 'hdf5'

local mnist_file = hdf5.open('mnist.hdf5', 'r')
local train_image = mnist_file:read('train_image'):all()
local train_label = mnist_file:read('train_label'):all()

local valid_image = mnist_file:read('valid_image'):all()
local valid_label = mnist_file:read('valid_label'):all()

train_set = {
    size = 50000,
    data = train_image[{{1, 50000}}]:double()
}

function model_make_layer()
    model = nn.Sequential()
    model:add(nn.Reshape(28, 28))
    model:add(nn.Tanh())
    model:add(nn.LogSoftMax())
    return model
end

function main()
    model = model_make_layer()
    input = torch.rand(1, 28, 28)
    model:forward(input)
    -- model:forward(train_set.data[1])
end

main()
