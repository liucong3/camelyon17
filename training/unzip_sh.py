import os
for root, folders, files in os.walk('.'):
    for file in files:
        if file.endswith('.zip'):
            zipfile = os.path.join(root, file)
            if not file.startswith('._'):
                print('unzip %s -d tif' % zipfile)
            print('rm %s' % zipfile)