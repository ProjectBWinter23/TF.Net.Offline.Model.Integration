using System.Drawing;
using System.IO;
using Tensorflow;
using Tensorflow.NumPy;


namespace RecyclingModel
{
    /// <summary>
    /// A class that is used to open and use a .tflite and .pd image classification models.
    /// </summary>
    public static class Accessor
    {
        public static void LoadModel()
        {
            // Instantiate a new Graph (an object by which the model is loaded into)
            Graph graph = new Graph();

            // Import trained model (.tflite) trial(1)
            string dir = "model.tflite";

            // Import trained model (.pd) trial(2)
            string dir2 = "saved_model.pb";
            string output_graph = dir;
            graph.Import(output_graph, "");

            // Load image from file
            string file_name = @"C:\Users\Marwa\Downloads\test3.jpg";

            // Resizing and normilizing the image in yeild a NDArray
            NDArray inputTensor = ReadTensorFromImageFile(file_name);

            Tensor input_operation = graph.OperationByName("sequential_1_input");
            Tensor output_operation = graph.OperationByName("outputs");

            NDArray result;

            // Run the model
            using (var sess = Tensorflow.Binding.tf.Session(graph))
            {
                result = sess.run(output_operation.outputs[0], new FeedItem(input_operation.outputs[0], inputTensor));
            }
        }

        private static NDArray ReadTensorFromImageFile(string file_name,
                                        int input_height = 256,
                                        int input_width = 256,
                                        int input_mean = 256,
                                        int input_std = 1)
        {
            var graph = Tensorflow.Binding.tf.Graph().as_default();

            var file_reader = Tensorflow.Binding.tf.io.read_file(file_name, "file_reader");
            var decodeJpeg = Tensorflow.Binding.tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            var cast = Tensorflow.Binding.tf.cast(decodeJpeg, Tensorflow.Binding.tf.float32);
            var dims_expander = Tensorflow.Binding.tf.expand_dims(cast, 0);
            var resize = Tensorflow.Binding.tf.constant(new int[] { input_height, input_width });
            var bilinear = Tensorflow.Binding.tf.image.resize_bilinear(dims_expander, resize);
            var sub = Tensorflow.Binding.tf.subtract(bilinear, new float[] { input_mean });
            var normalized = Tensorflow.Binding.tf.divide(sub, new float[] { input_std });

            using (var sess = Tensorflow.Binding.tf.Session(graph))
                return sess.run(normalized);

        }

       
    }
}
