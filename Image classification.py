import xgboost as xgb
import brightness_OpenCV
import PSPNet
import clarity_OpenCV
import detectvps
from pandas.core.accessor import PandasDelegate


imagepath =r'.\picture\test.jpg'
fig_savepath=r'.\picture'
name = 'test'
brightness = brightness_OpenCV.image_v (imagepath)
clarity = clarity_OpenCV.getImageVar(imagepath)


VPdata = detectvps.vpdetect (fig_savepath,name, imagepath)
VP_distance = VPdata.iloc[7]
VP_horizontal=VPdata.iloc[5]
VP_vertical = VPdata.iloc[6]
seg_ratio = PSPNet.get_seg(imagepath)


test_data = [brightness, VP_distance,clarity_OpenCV] + seg_ratio +[VP_horizontal,VP_vertical]
model = xgb.XGBClassifier()
model.load_model(r'.\imagefiltering_XGBoost_Zheng.model')

predictions = model.predict (test_data)
print (predictions)



