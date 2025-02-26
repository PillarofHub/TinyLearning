class PairGenTest:
    def __init__(self, X_support, y_support, X_test, y_test, batch_size=32, step=256, positive_label=1.0, negative_label=0.0):
        import numpy as np
        import tensorflow as tf
        
        input_shape = np.shape(X_support)
        
        # create indices of classes
        from collections import defaultdict
        support_idx = defaultdict(list)
        for y_support_idx, y in enumerate(y_support):
            support_idx[y].append(y_support_idx)

        test_idx = defaultdict(list)
        for y_test_idx, y in enumerate(y_test):
            test_idx[y].append(y_test_idx)
            
        keyList = []
        for key in support_idx:
            keyList.append(key)
                
        def generator_test():
            step_cnt = 0
            while step_cnt < step:
                step_cnt += 1
                batch_cnt = 0
                image_left = []
                image_right = []
                label = []
                
                while batch_cnt < batch_size:
                    batch_cnt += 1
                    current_label = np.random.randint(2)
                    if current_label == 1:
                        classNum = keyList[np.random.randint(len(test_idx))]
                        indexArr = [np.random.randint(len(support_idx[classNum])), np.random.randint(len(test_idx[classNum]))]
                        image_left.append(X_support[support_idx[classNum][indexArr[0]]])
                        image_right.append(X_test[test_idx[classNum][indexArr[1]]])
                        label.append([positive_label])
                    else:
                        classNumIndex = np.random.choice(len(test_idx), 2, replace=False)
                        classNum = [keyList[classNumIndex[0]], keyList[classNumIndex[1]]]
                        indexArr = [np.random.randint(len(support_idx[classNum[0]])), np.random.randint(len(test_idx[classNum[1]]))]
                        image_left.append(X_support[support_idx[classNum[0]][indexArr[0]]])
                        image_right.append(X_test[test_idx[classNum[1]][indexArr[1]]])
                        label.append([negative_label])
                        
                yield np.array([image_left, image_right]), np.array(label)
                
        self.ds_test = tf.data.Dataset.from_generator(generator_test, output_signature=(
            tf.TensorSpec(shape=(2, None, input_shape[1], input_shape[2], input_shape[3]), dtype=tf.float32), 
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        ))