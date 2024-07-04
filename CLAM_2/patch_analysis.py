from stardist.models import StarDist2D, StarDist3D
StarDist2D.from_pretrained()
model = StarDist2D.from_pretrained('2D_versatile_he')
# model = StarDist2D(config, name = 'mymodel')
labels, _ = model.predict_instances('/home/Drivessd2tb/Mohit_Combined/high_attention_patches_from_heatmap/ABC/000104CZ__20240628_091716/0.png')