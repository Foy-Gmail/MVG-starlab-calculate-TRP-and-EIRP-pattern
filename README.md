<img width="374" height="464" alt="image" src="https://github.com/user-attachments/assets/f925ce79-2730-427f-a85e-b3e96d99f163" />
1.AMS-8800 Series Antenna Measurement System
  
这个OTA chamber用来验证Iphone Antenna Active and Passive data
1.Iphone Antenna Active 使用CMW500和NRP和EMCenter信令测试 11.ax wifi 和 cell的TRP power。使用FSV ，NRP 和EMCenter来测试wifi BT Thread 非信令测试TRP power。
测试原理： 
phi轴 (Azimuth)：0,15,30,45,…,345（共 24 个点）。
theta轴 (Elevation)：0,15,30,45,…,165（共 12 个点）。
总采样点数：24x12 = 288个空间交叉点, 每个交点测量 H 极化 和 V 极化 EIRP，共 2 × 288 = 576 次测量
2.Iphone Passive 使用ZNB来测试S11。使用ZNB+EMCenter来控制AMS-8800测试Efficiency。
Acitve：
Gain=EIRP-conducted
Efficiency = TRP-Conducted
Directory = Gain-Efficiency = EIRP-TRP
Passive：
Efficiency use ZNB+chamber to measure
<img width="468" height="655" alt="image" src="https://github.com/user-attachments/assets/9ee5c33d-4353-4a5b-ba5c-1750562d4c5f" />

MVG测试TRP原理
<img width="1514" height="807" alt="image" src="https://github.com/user-attachments/assets/41b56100-735d-48ff-9381-ba1906ed5b41" />
<img width="1514" height="807" alt="image" src="https://github.com/user-attachments/assets/de36ff32-ed35-4716-be0b-2501e875ae17" />


对 RF 工程师的实际意义：
天线性能评估
TRP 告诉你这个设备实际能发出多少功率到空中。即使芯片输出功率达标，如果天线效率差、机身遮挡严重、馈线损耗大，TRP 就会偏低，用户实际通话质量和数据速率都会受影响。
定位问题方向：
3D 方向图能告诉你哪个方向功率低。比如手握持方向的 EIRP 明显低，说明手对那个方向的天线遮挡严重，需要调整天线位置或极化方向。

这个软件的意义就是把测出来的每个球点上的EIRP积分算出TRP，然后再看3D pattern看归一化EIRP图来判断辐射增益
这个是rawdata
<img width="1312" height="985" alt="image" src="https://github.com/user-attachments/assets/09f34382-5884-48a8-88b9-7eb7fef848ef" />
然后我们观看这个EIRP归一化辐射图
<img width="972" height="844" alt="Screenshot 2026-03-16 at 22 26 06" src="https://github.com/user-attachments/assets/1d5813c2-b9c2-47ef-b9d8-db89b94013e4" />
