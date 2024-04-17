python train_dual.py FordA  edf --loader HAR --batch-size 32 --data_perc 5perc
python train_tri.py HAR edf --loader HAR --batch-size 512 --data_perc 5perc
# Dataset: HAR
# Arguments: Namespace(dataset='HAR', run_name='edf', loader='HAR', gpu=0, batch_size=512, lr=0.001, repr_dims=320, max_train_length=3000, iters=None, epochs=40, save_every=None, seed=None, max_threads=None, eval=False, warmup=0.2, prototype_lambda=0.1, irregular=0, data_perc='5perc', ma_gamma=0.9999, temperature=0.1, model_path=None)
# =============================================
# Dataset: HAR
# Mode:    edf
# =============================================
# Namespace(dataset='HAR', run_name='edf', loader='HAR', gpu=0, batch_size=512, lr=0.001, repr_dims=320, max_train_length=3000, iters=None, epochs=40, save_every=None, seed=None, max_threads=None, eval=False, warmup=0.2, prototype_lambda=0.1, irregular=0, data_perc='5perc', ma_gamma=0.9999, temperature=0.1, model_path=None)
# Loading data... train
# /workspace/CA-TCC/data/HAR/
# torch.Size([5881, 128, 9])
# <class 'torch.Tensor'>
# torch.Size([5881, 128, 9])
# train_X.shape torch.Size([5881, 128, 9])
# train_X_fft.shape torch.Size([5881, 128, 9])
# train_X_sea.shape torch.Size([5881, 128, 9])
# done
# train_data[0] torch.Size([5881, 128, 9])
# /workspace/reference/ts2vec/ts2vec_tri.py:108: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at ../aten/src/ATen/native/IndexingUtils.h:27.)
#   train_data[0] = train_data[0][~np.isnan(train_data[0]).all(axis=2).all(axis=1)]
# /workspace/reference/ts2vec/ts2vec_tri.py:109: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at ../aten/src/ATen/native/IndexingUtils.h:27.)
#   train_data[1] = train_data[1][~np.isnan(train_data[1]).all(axis=2).all(axis=1)]
# Epoch #0: loss=27.096805572509766
# Epoch #1: loss=14.903123595497824
# Epoch #2: loss=13.810375907204367
# Epoch #3: loss=13.812053246931596
# Epoch #4: loss=13.155678402293812
# Epoch #5: loss=12.80215766213157
# Epoch #6: loss=12.40178324959495
# Epoch #7: loss=12.374310320073908
# Epoch #8: loss=12.161471366882324
# Epoch #9: loss=11.731566949324174
# Epoch #10: loss=11.395021438598633
# Epoch #11: loss=11.216839010065252
# Epoch #12: loss=11.248322226784445
# Epoch #13: loss=10.85983571139249
# Epoch #14: loss=10.689965508200906
# Epoch #15: loss=10.879681240428578
# Epoch #16: loss=10.124957604841752
# Epoch #17: loss=10.094717025756836
# Epoch #18: loss=10.229415633461691
# Epoch #19: loss=10.16800646348433
# ====================================================================================================
# tensor([3, 4, 4,  ..., 0, 2, 4], device='cuda:0')
# tensor([4, 0, 0,  ..., 3, 4, 0], device='cuda:0')
# tensor([4, 5, 5,  ..., 3, 4, 5], device='cuda:0')
# Epoch #20: loss=9.97250071438876
# Epoch #21: loss=10.879190878434615
# Epoch #22: loss=10.235125454989346
# Epoch #23: loss=10.156383947892623
# Epoch #24: loss=10.01305189999667
# Epoch #25: loss=9.751113371415572
# Epoch #26: loss=9.667603839527477
# Epoch #27: loss=9.26339071447199
# Epoch #28: loss=9.79831244728782
# Epoch #29: loss=9.460800257596103
# Epoch #30: loss=9.263521194458008
# Epoch #31: loss=9.677260052074086
# Epoch #32: loss=9.343487392772328
# Epoch #33: loss=8.98912577195601
# Epoch #34: loss=8.74344843084162
# Epoch #35: loss=8.520977540449662
# Epoch #36: loss=8.431622331792658
# Epoch #37: loss=7.9669717008417305
# Epoch #38: loss=8.18538522720337
# Epoch #39: loss=8.980508457530629

# Training time: 0:05:29.648342

# 5perc
# /workspace/CA-TCC/data/HAR/
# torch.Size([294, 128, 9])
# <class 'torch.Tensor'>
# torch.Size([294, 128, 9])
# train_X.shape torch.Size([294, 128, 9])
# train_X_fft.shape torch.Size([294, 128, 9])
# train_X_sea.shape torch.Size([294, 128, 9])
# ================== result ======================
# ==== acc : 0.9284017645062775
# ==== mf1 : 0.9283376674678502
# ==== roc : 0.9927198253668154
# ================== result ======================
# Evaluation result: {'acc': 0.9284017645062775, 'mf1': 0.9283376674678502, 'roc': 0.9927198253668154}
# Finished.
python train_dual.py FordA  edf --loader HAR --batch-size 128 --data_perc 5perc --gpu 2
# Dataset: FordA
# Arguments: Namespace(dataset='FordA', run_name='edf', loader='HAR', gpu=2, batch_size=128, lr=0.001, repr_dims=320, max_train_length=3000, iters=None, epochs=40, save_every=None, seed=None, max_threads=None, eval=False, warmup=0.5, irregular=0, data_perc='5perc', model_path=None, ma_gamma=0.9999, temperature=0.1)
# =============================================
# Dataset: FordA
# Mode:    edf
# =============================================
# Namespace(dataset='FordA', run_name='edf', loader='HAR', gpu=2, batch_size=128, lr=0.001, repr_dims=320, max_train_length=3000, iters=None, epochs=40, save_every=None, seed=None, max_threads=None, eval=False, warmup=0.5, irregular=0, data_perc='5perc', model_path=None, ma_gamma=0.9999, temperature=0.1)
# Loading data... train
# /workspace/CA-TCC/data/FordA/
# torch.Size([2880, 500, 1])
# <class 'torch.Tensor'>
# torch.Size([2880, 500, 1])
# train_X.shape torch.Size([2880, 500, 1])
# train_X_fft.shape torch.Size([2880, 500, 1])
# done
# train_data[0] torch.Size([2880, 500, 1])
# /workspace/reference/ts2vec/ts2vec_dual.py:98: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at ../aten/src/ATen/native/IndexingUtils.h:27.)
#   train_data[0] = train_data[0][~np.isnan(train_data[0]).all(axis=2).all(axis=1)]
# Epoch #0: loss=18.56381672078913
# Epoch #0: loss=18.56381672078913
# Epoch #1: loss=8.619367946277965
# Epoch #1: loss=8.619367946277965
# Epoch #2: loss=8.276834206147628
# Epoch #2: loss=8.276834206147628
# Epoch #3: loss=7.906106818806041
# Epoch #3: loss=7.906106818806041
# Epoch #4: loss=7.464538184079257
# Epoch #4: loss=7.464538184079257
# Epoch #5: loss=6.835667393424294
# Epoch #5: loss=6.835667393424294
# Epoch #6: loss=6.089054519479925
# Epoch #6: loss=6.089054519479925
# Epoch #7: loss=6.148156642913818
# Epoch #7: loss=6.148156642913818
# Epoch #8: loss=5.709125432101163
# Epoch #8: loss=5.709125432101163
# Epoch #9: loss=5.700544357299805
# Epoch #9: loss=5.700544357299805
# Epoch #10: loss=5.776944225484675
# Epoch #10: loss=5.776944225484675
# Epoch #11: loss=5.536010503768921
# Epoch #11: loss=5.536010503768921
# Epoch #12: loss=5.349580201235685
# Epoch #12: loss=5.349580201235685
# Epoch #13: loss=5.6197380586103955
# Epoch #13: loss=5.6197380586103955
# Epoch #14: loss=5.18195069919933
# Epoch #14: loss=5.18195069919933
# Epoch #15: loss=5.233273094350642
# Epoch #15: loss=5.233273094350642
# Epoch #16: loss=4.758063294670799
# Epoch #16: loss=4.758063294670799
# Epoch #17: loss=4.609305013309825
# Epoch #17: loss=4.609305013309825
# Epoch #18: loss=4.155880678783763
# Epoch #18: loss=4.155880678783763
# Epoch #19: loss=4.071756677194075
# Epoch #19: loss=4.071756677194075
# Epoch #20: loss=4.322517395019531
# Epoch #20: loss=4.322517395019531
# Epoch #21: loss=4.316570953889326
# Epoch #21: loss=4.316570953889326
# Epoch #22: loss=4.140321243893016
# Epoch #22: loss=4.140321243893016
# Epoch #23: loss=4.342283649878069
# Epoch #23: loss=4.342283649878069
# Epoch #24: loss=4.665197849273682
# Epoch #24: loss=4.665197849273682
# Epoch #25: loss=4.083278612657026
# Epoch #25: loss=4.083278612657026
# Epoch #26: loss=3.8874556584791704
# Epoch #26: loss=3.8874556584791704
# Epoch #27: loss=3.9947888309305366
# Epoch #27: loss=3.9947888309305366
# Epoch #28: loss=3.841365326534618
# Epoch #28: loss=3.841365326534618
# Epoch #29: loss=4.03315269947052
# Epoch #29: loss=4.03315269947052
# Epoch #30: loss=4.461756706237793
# Epoch #30: loss=4.461756706237793
# Epoch #31: loss=3.539191657846624
# Epoch #31: loss=3.539191657846624
# Epoch #32: loss=3.889615058898926
# Epoch #32: loss=3.889615058898926
# Epoch #33: loss=3.778295321898027
# Epoch #33: loss=3.778295321898027
# Epoch #34: loss=4.390855680812489
# Epoch #34: loss=4.390855680812489
# Epoch #35: loss=3.611179904504256
# Epoch #35: loss=3.611179904504256
# Epoch #36: loss=3.4388204379515215
# Epoch #36: loss=3.4388204379515215
# Epoch #37: loss=3.2937473383816807
# Epoch #37: loss=3.2937473383816807
# Epoch #38: loss=3.405354868281971
# Epoch #38: loss=3.405354868281971
# Epoch #39: loss=3.017758911306208
# Epoch #39: loss=3.017758911306208

# Training time: 0:08:38.150285

# 5perc
# /workspace/CA-TCC/data/FordA/
# torch.Size([144, 500, 1])
# <class 'torch.Tensor'>
# torch.Size([144, 500, 1])
# train_X.shape torch.Size([144, 500, 1])
# train_X_fft.shape torch.Size([144, 500, 1])
# ================== result ======================
# ==== acc : 0.9265151515151515
# ==== mf1 : 0.92642964878259
# ==== roc : 0.9788927725268235
# ================== result ======================
# Evaluation result: {'acc': 0.9265151515151515, 'mf1': 0.92642964878259, 'roc': 0.9788927725268235}
# Finished.

python train_dual.py StarLightCurves  edf --loader HAR --batch-size 64 --data_perc 5perc --gpu 1
# Dataset: StarLightCurves
# Arguments: Namespace(dataset='StarLightCurves', run_name='edf', loader='HAR', gpu=1, batch_size=64, lr=0.001, repr_dims=320, max_train_length=3000, iters=None, epochs=40, save_every=None, seed=None, max_threads=None, eval=False, warmup=0.5, irregular=0, data_perc='5perc', model_path=None, ma_gamma=0.9999, temperature=0.1)
# =============================================
# Dataset: StarLightCurves
# Mode:    edf
# =============================================
# Namespace(dataset='StarLightCurves', run_name='edf', loader='HAR', gpu=1, batch_size=64, lr=0.001, repr_dims=320, max_train_length=3000, iters=None, epochs=40, save_every=None, seed=None, max_threads=None, eval=False, warmup=0.5, irregular=0, data_perc='5perc', model_path=None, ma_gamma=0.9999, temperature=0.1)
# Loading data... train
# /workspace/CA-TCC/data/StarLightCurves/
# torch.Size([800, 1024, 1])
# <class 'torch.Tensor'>
# torch.Size([800, 1024, 1])
# train_X.shape torch.Size([800, 1024, 1])
# train_X_fft.shape torch.Size([800, 1024, 1])
# done
# train_data[0] torch.Size([800, 1024, 1])
# /workspace/reference/ts2vec/ts2vec_dual.py:98: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at ../aten/src/ATen/native/IndexingUtils.h:27.)
#   train_data[0] = train_data[0][~np.isnan(train_data[0]).all(axis=2).all(axis=1)]
# Epoch #0: loss=17.01677481333415
# Epoch #0: loss=17.01677481333415
# Epoch #1: loss=8.716346184412638
# Epoch #1: loss=8.716346184412638
# Epoch #2: loss=8.028510967890421
# Epoch #2: loss=8.028510967890421
# Epoch #3: loss=7.83284862836202
# Epoch #3: loss=7.83284862836202
# Epoch #4: loss=7.620809038480123
# Epoch #4: loss=7.620809038480123
# Epoch #5: loss=7.6607160568237305
# Epoch #5: loss=7.6607160568237305
# Epoch #6: loss=7.419819593429565
# Epoch #6: loss=7.419819593429565
# Epoch #7: loss=7.397915323575337
# Epoch #7: loss=7.397915323575337
# Epoch #8: loss=6.95219345887502
# Epoch #8: loss=6.95219345887502
# Epoch #9: loss=7.085602760314941
# Epoch #9: loss=7.085602760314941
# Epoch #10: loss=6.988589962323506
# Epoch #10: loss=6.988589962323506
# Epoch #11: loss=7.153656005859375
# Epoch #11: loss=7.153656005859375
# Epoch #12: loss=7.048841993014018
# Epoch #12: loss=7.048841993014018
# Epoch #13: loss=6.680593252182007
# Epoch #13: loss=6.680593252182007
# Epoch #14: loss=6.791816671689351
# Epoch #14: loss=6.791816671689351
# Epoch #15: loss=7.011043628056844
# Epoch #15: loss=7.011043628056844
# Epoch #16: loss=7.013467748959859
# Epoch #16: loss=7.013467748959859
# Epoch #17: loss=6.407669305801392
# Epoch #17: loss=6.407669305801392
# Epoch #18: loss=6.698480725288391
# Epoch #18: loss=6.698480725288391
# Epoch #19: loss=6.6407976150512695
# Epoch #19: loss=6.6407976150512695
# Epoch #20: loss=6.262795488039653
# Epoch #20: loss=6.262795488039653
# Epoch #21: loss=7.224381526311238
# Epoch #21: loss=7.224381526311238
# Epoch #22: loss=7.659167488416036
# Epoch #22: loss=7.659167488416036
# Epoch #23: loss=6.808157205581665
# Epoch #23: loss=6.808157205581665
# Epoch #24: loss=8.376445651054382
# Epoch #24: loss=8.376445651054382
# Epoch #25: loss=6.757030725479126
# Epoch #25: loss=6.757030725479126
# Epoch #26: loss=6.302858392397563
# Epoch #26: loss=6.302858392397563
# Epoch #27: loss=6.801898558934529
# Epoch #27: loss=6.801898558934529
# Epoch #28: loss=6.582660675048828
# Epoch #28: loss=6.582660675048828
# Epoch #29: loss=6.419068853060405
# Epoch #29: loss=6.419068853060405
# Epoch #30: loss=6.128835995992024
# Epoch #30: loss=6.128835995992024
# Epoch #31: loss=6.184444228808085
# Epoch #31: loss=6.184444228808085
# Epoch #32: loss=6.0415955781936646
# Epoch #32: loss=6.0415955781936646
# Epoch #33: loss=6.165836016337077
# Epoch #33: loss=6.165836016337077
# Epoch #34: loss=6.373576958974202
# Epoch #34: loss=6.373576958974202
# Epoch #35: loss=6.085099697113037
# Epoch #35: loss=6.085099697113037
# Epoch #36: loss=6.3741455078125
# Epoch #36: loss=6.3741455078125
# Epoch #37: loss=6.085709611574809
# Epoch #37: loss=6.085709611574809
# Epoch #38: loss=5.899378418922424
# Epoch #38: loss=5.899378418922424
# Epoch #39: loss=6.426173806190491
# Epoch #39: loss=6.426173806190491

# Training time: 0:05:39.794689

# 5perc
# /workspace/CA-TCC/data/StarLightCurves/
# torch.Size([40, 1024, 1])
# <class 'torch.Tensor'>
# torch.Size([40, 1024, 1])
# train_X.shape torch.Size([40, 1024, 1])
# train_X_fft.shape torch.Size([40, 1024, 1])
# ================== result ======================
# ==== acc : 0.9362554638173871
# ==== mf1 : 0.8923458903919127
# ==== roc : 0.9821687196283064
# ================== result ======================
# Evaluation result: {'acc': 0.9362554638173871, 'mf1': 0.8923458903919127, 'roc': 0.9821687196283064}
# Finished.