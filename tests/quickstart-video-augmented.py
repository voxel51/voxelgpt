DSETNAME = 'quickstart-video-augmented'
if fo.dataset_exists(DSETNAME):
    fo.delete_dataset(DSETNAME)
    
dsv = foz.load_zoo_dataset('quickstart-video',dataset_name=DSETNAME)
dsv.compute_metadata()
dsv.ensure_frames()
dsv.persistent=True

dets = dsv.values('frames.detections')
dsv.set_values('frames.dets_mnet',dets)

dsv.delete_frame_field('frames.detections')

# tags
set = np.random.randint(0,2,len(dsv),dtype=np.bool)
dsv[set].tag_samples('Intruder')
set = np.random.randint(0,2,len(dsv),dtype=np.bool)
dsv[set].tag_samples('Safe')

# classes
CLASSES = ['sunny', 'rainy', 'hailing']
num_class = len(CLASSES)
for s in dsv.iter_samples(autosave=True):
    icls = np.random.randint(0,num_class)
    conf = np.random.random()
    c = fo.Classification(label=CLASSES[icls],confidence=conf)
    s['weather'] = c
    
# tempdets
CLASSES = ['vehicle','person','animal']
for s in dsv.iter_samples(autosave=True):
    nf = s.metadata.total_frame_count
    icls = np.random.randint(0,num_class)
    conf = np.random.random()
    f0 = np.random.randint(0,nf)
    f1 = np.random.randint(0,nf)
    if f0 > f1:
        tmp = f0
        f0 = f1
        f1 = tmp
    s['objects'] = fo.TemporalDetection(label=CLASSES[icls],
                                        support=[f0,f1],
                                        confidence=conf)
    
# frame classes
for s in dsv.iter_samples(autosave=True):
    for f,frm in s.frames.items():
        if np.random.random()>0.95:
            icls = np.random.randint(0,num_class)
            conf = np.random.random()
            c = fo.Classification(label=CLASSES[icls],confidence=conf)
            frm['intruder'] = c
            frm.save()
                                    
    
