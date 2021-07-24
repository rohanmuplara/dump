def get_datasetBOTTOMS(opts, split):
    global gridImg, mapping, armmapping, targetClothClasses
	gridImg = tf.cast(tf.image.decode_png(tf.io.read_file(join('gs://experiments_logs/datasets', 'grid.png'))), tf.float32)
	gridImg = (gridImg/255. - [0.5, 0.5, 0.5])/[0.5,0.5,0.5]
	mapping = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 13, 13, 8, 9, 13, 10, 11, 12, 13, 13, 13, 13])
	targetClothClasses = [5, 6, 7] if "Tops" in opts.dataset else [9, 12]
	armmapping = [0, 1, 2, 1, 2, 3]
    featureDescription = {
	'cloth' : tf.io.FixedLenFeature([], tf.string),
	'clothMask' : tf.io.FixedLenFeature([], tf.string),

	'warpedCloth' : tf.io.FixedLenFeature([], tf.string),
	'warpedClothMask' : tf.io.FixedLenFeature([], tf.string),

	'person' : tf.io.FixedLenFeature([], tf.string),

	'personMask' : tf.io.FixedLenFeature([], tf.string),
	'exppersonMask' : tf.io.FixedLenFeature([], tf.string),

	'densepose' : tf.io.FixedLenFeature([], tf.string),

	'personno' : tf.io.FixedLenFeature([], tf.int64),
	'clothno' : tf.io.FixedLenFeature([], tf.int64),

	dataset = dataset.map(lambda x : tf.io.parse_single_example(x, featureDescription))
     data['cloth'] = (tf.cast(tf.image.decode_jpeg(data['cloth'], channels=3), tf.float32)/255. - 0.5)/0.5 # HxWx3
	data['clothMask'] = tf.cast(tf.image.decode_jpeg(data['clothMask'], channels=1), tf.float32)/255. # HxWx3

	data['cloth'] = data['cloth']*data['clothMask'] + (1-data['clothMask'])

	data['warpedCloth'] = (tf.cast(tf.image.decode_jpeg(data['warpedCloth'], channels=3), tf.float32)/255. - 0.5)/0.5 # HxWx3
	data['warpedClothMask'] = tf.cast(tf.image.decode_jpeg(data['warpedClothMask'], channels=1)[:,:,0], tf.float32)/255. # HxWx3

	data['person'] = (tf.cast(tf.image.decode_jpeg(data['person'], channels=3), tf.float32)/255. - 0.5)/0.5 # HxWx3

	data['personMask'] = tf.image.decode_png(data['personMask'], channels=1)[:,:,0]
	data['personMask'] = tf.cast(data['personMask'], tf.int64) # HxW
	#data['personMask'] = tf.gather(mapping, data['personMask'])

	data['shapeMask'] = tf.cast(data['personMask'] > 0, tf.dtypes.float32) # HxW

	data['exppersonMask'] = tf.image.decode_png(data['exppersonMask'], channels=1)[:,:,0]
	data['exppersonMask'] = tf.cast(data['exppersonMask'], tf.int64) # HxW
	#data['exppersonMask'] = tf.gather(mapping, data['exppersonMask'])

	data['personClothMask'] = filterMask(data['personMask'], targetClothClasses)
	data['personCloth'] = data['person'] * data['personClothMask'][:,:,None] + (1 - data['personClothMask'][:,:,None])

	#data['inpaint'] = tf.cast(tf.image.decode_png(data['inpaint'], channels=1), tf.float32)/255.

	data['targetClothMask'] = filterMask(data['exppersonMask'][:,:,None], targetClothClasses)

	densepose = tf.image.decode_png(data['densepose'], channels=3)
	uv = (tf.cast(densepose[:,:,1:], tf.dtypes.float32)/255. - 0.5)/0.5
	densepose = B.concatenate([tf.cast(tf.one_hot(densepose[:,:,0], 25), tf.dtypes.float32), uv], -1)
	data['densepose'] = densepose

	#data['clotharmseg'] = tf.cast(tf.image.decode_png(data['clotharmseg'], channels=1), tf.int32)
	#data['personclotharmseg'] = tf.cast(tf.image.decode_png(data['personclotharmseg'], channels=1), tf.int32)
	#data['personclotharmseg'] = tf.gather(armmapping, data['personclotharmseg'])

	data['gridImg'] = gridImg
	data['gridImg'].set_shape([256,192,3])	


	temp = filterMask(data['personMask'], [16,17])
	case1 = filterMask(data['exppersonMask'],[5,6,7])[:,:,None]
	case2 = filterMask(data['exppersonMask'], [3,8,10,15,14,18,19])[:,:,None]

	x1 = B.argmax(B.reshape(temp, [256*192]))//192
	x2 = 256 - B.argmax(B.reshape(temp[::-1], [256*192]))//192

	diff = tf.cast(tf.random.uniform(shape=())*tf.cast(x2-x1, tf.float32), tf.int64)
	randomask = B.concatenate((tf.zeros([x1,192,1]), tf.zeros([x2-x1-diff, 192, 1]), tf.ones([diff, 192, 1]), tf.zeros([256-x2,192,1])), 0)
	#data['limbrandomask'] = randomask

	
	x1 = B.argmax(B.reshape(case1, [256*192]))//192
	x2 = 256 - B.argmax(B.reshape(case1[::-1], [256*192]))//192
	diff = tf.cast((1-(0.1+tf.random.uniform(shape=())*0.15))*tf.cast(x2-x1, tf.float32), tf.int64)
	randomask = B.concatenate((tf.zeros([x1,192,1]), tf.ones([diff, 192, 1]), tf.zeros([x2-x1-diff, 192, 1]), tf.zeros([256-x2,192,1])), 0)
	case1 = case1*randomask
	data['grapy'] = tf.zeros([256, 192], tf.int32)
data = defineBottomsInputs()
    return data, data['cloth'], data['clothMask'], tf.expand_dims(data['personClothMask'], axis=3),  data['gridImg']
 data, cloth, clothMask, personClothMask, gridImg = flowinput()
  out = singleTPS(cloth, clothMask, personClothMask, gridImg, opts)
  {cloth = cloth * clothMask + (1 - clothMask)
    source = B.concatenate([cloth, clothMask], -1)
    target = personClothMask
    tpsgrid = gmmtps(opts, target, source)

    tpsCloth = grid_sample(cloth, tpsgrid, cloth[:, 0, 0])
    tpsClothMask = grid_sample(clothMask, tpsgrid, clothMask[:, 0, 0])
    tpsWarpGrid = grid_sample(gridImg, tpsgrid, tf.ones([1, 3]) * 0.5)
    return {
        "warpedCloth": tpsCloth,
        "warpedMask": tpsClothMask,
        "warpedGrid": tpsWarpGrid,
    }
    }
  warpedCloth = out["warpedCloth"] * out["warpedMask"]

  output = {
    "warpedCloth": warpedCloth,
    "warpedMask": out["warpedMask"],
    "warpedGrid": out["warpedGrid"],
  }
	return dataset
