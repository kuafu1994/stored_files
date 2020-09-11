
import tensorflow as tf
import argparse
import time
import numpy as np
from tensorflow.saved_model import signature_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

parser = argparse.ArgumentParser()
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=1000)
parser.add_argument("--discard_iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

converter = trt.TrtGraphConverter(input_saved_model_dir='./resnet50',
				precision_mode='FP32')
converter.convert()

converter.save('./resnet50_trt_fp32')

randvalues = np.random.random_sample((1,64,56,56))

with tf.Session() as sess:
    # 
    times = []

    model = tf.saved_model.loader.load(sess, [tf.saved_model.SERVING],
				'./resnet50_trt_fp32')
    input_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['input'].name
    input = tf.get_default_graph().get_tensor_by_name(input_name)

    output_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['output'].name
    output = tf.get_default_graph().get_tensor_by_name(output_name)

    for i in range(args.discard_iter + args.iterations):
        t0 = time.time()
        sess.run([output], feed_dict={input: randvalues})
        t1 = time.time()
        times.append(t1 - t0)
    total = 0
    for i in range(args.discard_iter, len(times)):
        total += times[i]
    avg = total / (args.iterations) * 1000.0

    print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")
        
