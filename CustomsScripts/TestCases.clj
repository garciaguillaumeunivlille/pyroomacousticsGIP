track
safeMat = pra.Material(energy_absorption=1.0, scattering=0.0)

✅Mat OUT [IS][F 1-2]
- IR OK
LOG
  Done STL imports elapsed time : 0.0488 seconds
  Created Room elapsed time : 5.6998 seconds
  setup microphones elapsed time : 5.7001 seconds
  start main loop elapsed time : 5.7005 seconds
  begin image source elapsed time : 7.8342 seconds
  --ImageSource for source F-- elapsed time : 138.3988 seconds
  --Simulate sound with source F-- elapsed time : 138.4791 seconds
  Export F1.wav 1/2 elapsed time : 138.5579 seconds
  Export F2.wav 2/2 elapsed time : 138.5663 seconds
  Json Export elapsed time : 142.7911 seconds
   Total elapsed time : 142.7915 seconds
------------------------------------------------------------------------------------------

✅Mat OUT [IS,RT][F 1-2]
- image source OK
- RT OK
- IRs OK
LOG
  Room created elapsed time : 5.7528 seconds
  setup microphones elapsed time : 5.7728 seconds
  start main loop elapsed time : 5.7730 seconds
  begin image source elapsed time : 7.7852 seconds
  --ImageSource for source F-- elapsed time : 140.6461 seconds
  number of rays : 110896
  --RayTracing for source F-- elapsed time : 148.4391 seconds
  --Simulate sound with source F-- elapsed time : 148.5209 seconds
  Export F1.wav 1/2 elapsed time : 148.6000 seconds
  Export F2.wav 2/2 elapsed time : 148.6083 seconds
  Json Export elapsed time : 156.4017 seconds
   Total elapsed time : 156.4019 seconds
------------------------------------------------------------------------------------------

❌Mat IN [IS,RT][F 1-2]
- image source OK
- RT ERROR
LOG
Done STL imports elapsed time : 0.0506 seconds
Created Room elapsed time : 5.8004 seconds
setup microphones elapsed time : 5.8210 seconds
start main loop elapsed time : 5.8213 seconds
begin image source elapsed time : 7.7497 seconds
--ImageSource for source F-- elapsed time : 102.2695 seconds
Traceback (most recent call last):
  File "C:\Users\trogl\Documents\pyroomacousticsGIP\CustomsScripts\loop.py", line 346, in <module>
    TheatreRoom.ray_tracing()
  File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\room.py", line 2258, in ray_tracing
    self.room_engine.ray_tracing(self.rt_args["n_rays"], src.position)
TypeError: ray_tracing(): incompatible function arguments. The following argument types are supported:
    1. (self: pyroomacoustics.libroom.Room, arg0: numpy.ndarray[numpy.float32[2, n]], arg1: numpy.ndarray[numpy.float32[3, 1]]) -> None
    2. (self: pyroomacoustics.libroom.Room, arg0: int, arg1: int, arg2: numpy.ndarray[numpy.float32[3, 1]]) -> None
    3. (self: pyroomacoustics.libroom.Room, arg0: int, arg1: numpy.ndarray[numpy.float32[3, 1]]) -> None

Invoked with: <pyroomacoustics.libroom.Room object at 0x0000022B4A435CB0>, -110896, array([ 0. , -1. ,  1.4])
------------------------------------------------------------------------------------------

✅Mat IN [IS][F 1-2]
- image source OK
LOG
    Done STL imports elapsed time : 0.0499 seconds
    Created Room elapsed time : 5.7140 seconds
    setup microphones elapsed time : 5.7143 seconds
    start main loop elapsed time : 5.7146 seconds
    begin image source elapsed time : 7.8292 seconds
    --ImageSource for source F-- elapsed time : 101.3047 seconds
    --Simulate sound with source F-- elapsed time : 101.3820 seconds
    Export F1.wav 1/2 elapsed time : 101.4596 seconds
    Export F2.wav 2/2 elapsed time : 101.4679 seconds
    Json Export elapsed time : 107.8586 seconds
     Total elapsed time : 107.8590 seconds
------------------------------------------------------------------------------------------

 
❌Mat IN [IS][A-G 1-16]
- image source ERROR
LOG
    Done STL imports elapsed time : 0.0492 seconds
    Created Room elapsed time : 5.7014 seconds
    setup microphones elapsed time : 5.7022 seconds
    start main loop elapsed time : 5.7025 seconds
    begin image source elapsed time : 8.7780 seconds
    --ImageSource for source A-- elapsed time : 878.4326 seconds
    Traceback (most recent call last):
      File "C:\Users\trogl\Documents\pyroomacousticsGIP\CustomsScripts\loop.py", line 352, in <module>
        TheatreRoom.simulate()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\room.py", line 2645, in simulate
        self.compute_rir()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\room.py", line 2303, in compute_rir
        ir_ism = compute_ism_rir(
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\simulation\ism.py", line 309, in compute_ism_rir
        t_max = time.max()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\numpy\_core\_methods.py", line 44, in _amax
        return umr_maximum(a, axis, None, out, keepdims, initial, where)
    ValueError: zero-size array to reduction operation maximum which has no identity

❌Mat IN [IS][F 1-16]
- image source ERROR
LOG
    Done STL imports elapsed time : 0.0519 seconds
    Created Room elapsed time : 5.7798 seconds
    setup microphones elapsed time : 5.7805 seconds
    start main loop elapsed time : 5.7812 seconds
    begin image source elapsed time : 8.8879 seconds
    --ImageSource for source F-- elapsed time : 814.9812 seconds
    Traceback (most recent call last):
      File "C:\Users\trogl\Documents\pyroomacousticsGIP\CustomsScripts\loop.py", line 352, in <module>
        TheatreRoom.simulate()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\room.py", line 2645, in simulate
        self.compute_rir()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\room.py", line 2303, in compute_rir
        ir_ism = compute_ism_rir(
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\simulation\ism.py", line 309, in compute_ism_rir
        t_max = time.max()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\numpy\_core\_methods.py", line 44, in _amax
        return umr_maximum(a, axis, None, out, keepdims, initial, where)
    ValueError: zero-size array to reduction operation maximum which has no identity


❌Mat IN [IS][F ONE BY ONE 1-16] 141000

PASSED:
- 2,3,5,6,8,9,10,11,13,15,16
FAILED:
- 1,4,7,12
LOG
    Done STL imports elapsed time : 0.0519 seconds
    Created Room elapsed time : 5.7798 seconds
    setup microphones elapsed time : 5.7805 seconds
    start main loop elapsed time : 5.7812 seconds
    begin image source elapsed time : 8.8879 seconds
    --ImageSource for source F-- elapsed time : 814.9812 seconds
    Traceback (most recent call last):
      File "C:\Users\trogl\Documents\pyroomacousticsGIP\CustomsScripts\loop.py", line 352, in <module>
        TheatreRoom.simulate()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\room.py", line 2645, in simulate
        self.compute_rir()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\room.py", line 2303, in compute_rir
        ir_ism = compute_ism_rir(
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\pyroomacoustics\simulation\ism.py", line 309, in compute_ism_rir
        t_max = time.max()
      File "C:\Users\trogl\AppData\Local\Programs\Python\Python39\lib\site-packages\numpy\_core\_methods.py", line 44, in _amax
        return umr_maximum(a, axis, None, out, keepdims, initial, where)
    ValueError: zero-size array to reduction operation maximum which has no identity

❌Mat IN [IS][F [1-16] except [1,4,7,12]]
❌Mat OUT [IS][F [1-16] except [1,4,7,12]] 141000
ValueError: zero-size array to reduction operation maximum which has no identity


> tester IN sans RT

> relancer mêmes tests avec + d'energy_absorption
> relancer mêmes test avec RT 
> relancer même tests sur OUT 
