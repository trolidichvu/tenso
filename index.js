let tf = require('@tensorflow/tfjs-node');
let fs = require('fs');
// Define a model for linear regression.
const model = tf.sequential();
model.add(tf.layers.dense({ units: 3, inputShape: [3] }));
model.add(tf.layers.dense({ units: 3}));
model.add(tf.layers.dense({ units: 3}));
model.add(tf.layers.dense({ units: 1}));
// Prepare the model for training: Specify the loss and the optimizer.
model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });


// read data. 
let dataRaw = fs.readFileSync('./data.csv', 'utf-8').split('\n');

let data = { xTrain: [], yTrain: [] }
for (let i = 0; i < dataRaw.length - 1; i++) {
  let line = dataRaw[i].split(',');
  for (let i = 0; i < line.length; i++) line[i] = parseFloat(line[i])
  let output = [parseFloat(line.pop())]
  let input = line;
  data.xTrain.push(input);
  data.yTrain.push(output)
}
const xs = tf.tensor2d(data.xTrain, [
  data.xTrain.length,
  data.xTrain[0].length
]);
const ys = tf.tensor2d(data.yTrain, [
  data.yTrain.length,
  data.yTrain[0].length
]);
// Train the model using the data.
model.fit(xs, ys, { epochs: 1000 }).then(() => {
  model.predict(tf.tensor2d([3,3,3,9,9,9,100], [data.yTrain[0].length, data.xTrain[0].length])).print();
});

