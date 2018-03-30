
import os, random, csv

def generate_csv():
    id_range = list(range(100))
    random.shuffle(id_range)
    dev_id_range = id_range[0:10]
    test_id_range = id_range[10:20]
    # train_ids = [4,9,10]
    # dev_ids = [12]
    # test_ids = [15]
    train_n = []
    train_t = []
    dev_n = []
    dev_t = []
    test_n = []
    test_t = []
    for root, dirs, files in os.walk('patches'):
        if root.endswith('patches'):
            for file in files:
                is_t = file.startswith('t')
                is_n = file.startswith('n')
                if not (is_t or is_n): continue
                patient_id = int(root.split('_')[1])
                path = os.path.join(root, file)
                if patient_id in dev_id_range:
                    if is_t:
                        dev_t.append(path)
                    else:
                        dev_n.append(path)
                elif patient_id in test_id_range:
                    if is_t:
                        test_t.append(path)
                    else:
                        test_n.append(path)
                else:
                    if is_t:
                        train_t.append(path)
                    else:
                        train_n.append(path)
    random.shuffle(train_t)
    random.shuffle(train_n)
    random.shuffle(dev_t)
    random.shuffle(dev_n)
    random.shuffle(test_t)
    random.shuffle(test_n)
    print_file(train_t, train_n, 'train.csv')
    print_file(dev_t, dev_n, 'dev.csv')
    print_file(test_t, test_n, 'test.csv')

def print_file(train_t, train_n, path):
    n = min(len(train_t), len(train_n))
    print('%s : %d*2' % (path, n))
    train = [[x, 1] for x in train_t[0:n]] + [[x, 0] for x in train_n[0:n]]
    file = open(path, 'w', encoding='utf-8')
    writer = csv.writer(file)
    writer.writerows(train)
    file.close()
    

if __name__ == '__main__':
    generate_csv()