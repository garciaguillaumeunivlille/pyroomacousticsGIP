🔘✅❌
safeMat = pra.Material(energy_absorption=1.0, scattering=0.0)
idealMat = pra.Material(energy_absorption=0.01, scattering=0.8)

----------------------------------------------ImageSource only 2 Micros----------------------------------------------

✅SafeMat OUT [IS][F 1-2]
- IR OK
LOG
  Done STL imports elapsed time : 0.0317 seconds
  Created Room elapsed time : 2.7077 seconds
  setup microphones elapsed time : 2.7079 seconds
  start main loop elapsed time : 2.7081 seconds
  begin image source elapsed time : 3.7365 seconds
  --ImageSource for source F-- elapsed time : 101.2073 seconds
  --Simulate sound with source F-- elapsed time : 101.3154 seconds
  Export F1.wav 1/2 elapsed time : 101.3702 seconds
  Export F2.wav 2/2 elapsed time : 101.3809 seconds
  Json Export elapsed time : 101.4584 seconds
   Total elapsed time : 101.4608 seconds
-----------------------------------------

✅SafeMat IN [IS][F 1-2]
- IR ✅
LOG
  Done STL imports elapsed time : 0.0700 seconds
  Created Room elapsed time : 3.3953 seconds
  setup microphones elapsed time : 3.3956 seconds
  start main loop elapsed time : 3.3957 seconds
  begin image source elapsed time : 4.7651 seconds
  --ImageSource for source F-- elapsed time : 73.4428 seconds
  --Simulate sound with source F-- elapsed time : 73.5472 seconds
  Export F1.wav 1/2 elapsed time : 73.5996 seconds
  Export F2.wav 2/2 elapsed time : 73.6123 seconds
  Json Export elapsed time : 73.6616 seconds
   Total elapsed time : 73.6619 seconds
-----------------------------------------

✅IdealMat OUT [IS][F 1-2]
- IR ✅
LOG
  Done STL imports elapsed time : 0.0285 seconds
  Created Room elapsed time : 2.6763 seconds
  setup microphones elapsed time : 2.6765 seconds
  start main loop elapsed time : 2.6766 seconds
  begin image source elapsed time : 3.6885 seconds
  --ImageSource for source F-- elapsed time : 110.6603 seconds
  --Simulate sound with source F-- elapsed time : 110.7514 seconds
  Export F1.wav 1/2 elapsed time : 110.8052 seconds
  Export F2.wav 2/2 elapsed time : 110.8117 seconds
  Json Export elapsed time : 110.8823 seconds
   Total elapsed time : 110.8843 seconds
-----------------------------------------

✅IdealMat IN [IS][F 1-2]
- IR ✅
LOG
  Done STL imports elapsed time : 0.0295 seconds
  Created Room elapsed time : 2.6895 seconds
  setup microphones elapsed time : 2.6897 seconds
  start main loop elapsed time : 2.6898 seconds
  begin image source elapsed time : 3.6859 seconds
  --ImageSource for source F-- elapsed time : 50.8069 seconds
  --Simulate sound with source F-- elapsed time : 50.8627 seconds
  Export F1.wav 1/2 elapsed time : 50.8951 seconds
  Export F2.wav 2/2 elapsed time : 50.9000 seconds
  Json Export elapsed time : 50.9401 seconds
   Total elapsed time : 50.9403 seconds

----------------------------------------------All Micros

🔘SafeMat OUT [IS][F 1-16]
- IR OK
LOG

-----------------------------------------

🔘SafeMat IN [IS][F 1-16]
- IR 🔘
LOG

-----------------------------------------

🔘IdealMat OUT [IS][F 1-16]
- IR 🔘
LOG

-----------------------------------------

❌IdealMat IN [IS][F 1-16]
Done STL imports elapsed time : 0.0310 seconds
Created Room elapsed time : 5.7689 seconds
setup microphones elapsed time : 5.7694 seconds
start main loop elapsed time : 5.7696 seconds
begin image source elapsed time : 9.0729 seconds
--ImageSource for source F-- elapsed time : 473.5821 seconds
LOG
  Traceback (most recent call last):
    File "C:\Users\Guillaume\pyroomacousticsGIP\CustomsScripts\loop.py", line 352, in <module>
      TheatreRoom.simulate() #TRACK 1/5
      ~~~~~~~~~~~~~~~~~~~~^^
    File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py", line 2645, in simulate
      self.compute_rir()
      ~~~~~~~~~~~~~~~~^^
    File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py", line 2303, in compute_rir
      ir_ism = compute_ism_rir(
          src,
      ...<8 lines>...
          min_phase=self.min_phase,
      )
    File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\simulation\ism.py", line 309, in compute_ism_rir
      t_max = time.max()
    File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\numpy\_core\_methods.py", line 43, in _amax
      return umr_maximum(a, axis, None, out, keepdims, initial, where)
  ValueError: zero-size array to reduction operation maximum which has no identity

----------------------------------------------ImageSource & RT----------------------------------------------

✅SafeMat OUT [IS][RT][F 1-2]
- IR ✅
LOG
  Done STL imports elapsed time : 0.0318 seconds
  C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py:1118: UserWarning: The number of rays used for ray tracing is larger than100000 which may result in slow simulation.  The numberof rays was automatically chosen to provide accurateroom impulse response based on the room volume and thereceiver radius around the microphones.  The number ofrays may be reduced by increasing the size of thereceiver.  This tends to happen especially for largerooms with small receivers.  The receiver is a spherearound the microphone and its radius (in meters) may bespecified by providing the `receiver_radius` keywordargument to the `set_ray_tracing` method.
    warnings.warn(
  Created Room elapsed time : 2.7295 seconds
  setup microphones elapsed time : 2.7387 seconds
  start main loop elapsed time : 2.7388 seconds
  begin image source elapsed time : 3.6771 seconds
  --ImageSource for source F-- elapsed time : 96.4445 seconds
  number of rays : 110896
  --RayTracing for source F-- elapsed time : 102.7690 seconds
  --Simulate sound with source F-- elapsed time : 102.8787 seconds
  Export F1.wav 1/2 elapsed time : 102.9346 seconds
  Export F2.wav 2/2 elapsed time : 102.9425 seconds
  Json Export elapsed time : 103.0246 seconds
   Total elapsed time : 103.0266 seconds
-----------------------------------------

❌SafeMat IN [IS][RT][F 1-2]
- IR ❌
LOG
  Done STL imports elapsed time : 0.0290 seconds
  Created Room elapsed time : 5.8936 seconds
  setup microphones elapsed time : 5.9228 seconds
  start main loop elapsed time : 5.9231 seconds
  begin image source elapsed time : 7.3256 seconds
  --ImageSource for source F-- elapsed time : 102.6298 seconds
  Traceback (most recent call last):
    File "C:\Users\Guillaume\pyroomacousticsGIP\CustomsScripts\loop.py", line 346, in <module>
      TheatreRoom.ray_tracing()
      ~~~~~~~~~~~~~~~~~~~~~~~^^
    File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py", line 2258, in ray_tracing
      self.room_engine.ray_tracing(self.rt_args["n_rays"], src.position)
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  TypeError: ray_tracing(): incompatible function arguments. The following argument types are supported:
      1. (self: pyroomacoustics.libroom.Room, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[2, n]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"]) -> None
      2. (self: pyroomacoustics.libroom.Room, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"]) -> None
      3. (self: pyroomacoustics.libroom.Room, arg0: typing.SupportsInt, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"]) -> None

  Invoked with: <pyroomacoustics.libroom.Room object at 0x000000006E384470>, -110896, array([ 0. , -1. ,  1.4])
-----------------------------------------

✅IdealMat OUT [IS][RT][F 1-2]
- IR ✅
LOG
  Done STL imports elapsed time : 0.0297 seconds
  C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py:1118: UserWarning: The number of rays used for ray tracing is larger than100000 which may result in slow simulation.  The numberof rays was automatically chosen to provide accurateroom impulse response based on the room volume and thereceiver radius around the microphones.  The number ofrays may be reduced by increasing the size of thereceiver.  This tends to happen especially for largerooms with small receivers.  The receiver is a spherearound the microphone and its radius (in meters) may bespecified by providing the `receiver_radius` keywordargument to the `set_ray_tracing` method.
    warnings.warn(
  Created Room elapsed time : 2.8168 seconds
  setup microphones elapsed time : 2.8260 seconds
  start main loop elapsed time : 2.8261 seconds
  begin image source elapsed time : 3.7970 seconds
  --ImageSource for source F-- elapsed time : 73.5948 seconds
  number of rays : 110896
  --RayTracing for source F-- elapsed time : 180.5832 seconds
  --Simulate sound with source F-- elapsed time : 180.7253 seconds
  Export F1.wav 1/2 elapsed time : 180.7772 seconds
  Export F2.wav 2/2 elapsed time : 180.7927 seconds
  Json Export elapsed time : 180.8347 seconds
   Total elapsed time : 180.8388 seconds
-----------------------------------------

❌IdealMat IN [IS][RT][F 1-2]
- IR ❌
LOG
  Done STL imports elapsed time : 0.0318 seconds
  Created Room elapsed time : 2.7795 seconds
  setup microphones elapsed time : 2.7894 seconds
  start main loop elapsed time : 2.7896 seconds
  begin image source elapsed time : 3.7305 seconds
  --ImageSource for source F-- elapsed time : 50.7019 seconds
  Traceback (most recent call last):
    File "C:\Users\Guillaume\pyroomacousticsGIP\CustomsScripts\loop.py", line 346, in <module>
      TheatreRoom.ray_tracing()
      ~~~~~~~~~~~~~~~~~~~~~~~^^
    File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py", line 2258, in ray_tracing
      self.room_engine.ray_tracing(self.rt_args["n_rays"], src.position)
      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  TypeError: ray_tracing(): incompatible function arguments. The following argument types are supported:
      1. (self: pyroomacoustics.libroom.Room, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[2, n]"], arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"]) -> None
      2. (self: pyroomacoustics.libroom.Room, arg0: typing.SupportsInt, arg1: typing.SupportsInt, arg2: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"]) -> None
      3. (self: pyroomacoustics.libroom.Room, arg0: typing.SupportsInt, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[3, 1]"]) -> None

  Invoked with: <pyroomacoustics.libroom.Room object at 0x000000006E2EFEB0>, -110896, array([ 0. , -1. ,  1.4])
------------------------------------------------------------------------------------------

- - - - - - - - - HIER

✅SafeMat OUT [IS,RT][F 1-2]
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

❌SafeMat IN [IS,RT][F 1-2]
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

✅SafeMat IN [IS][F 1-2]
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

 
❌SafeMat IN [IS][A-G 1-16]
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

❌SafeMat IN [IS][F 1-16]
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


❌SafeMat IN [IS][F ONE BY ONE 1-16] 141000

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
------------------------------------------------------------------------------------------

❌SafeMat IN [IS][F [1-16] except [1,4,7,12]]
❌SafeMat OUT [IS][F [1-16] except [1,4,7,12]] 141000
ValueError: zero-size array to reduction operation maximum which has no identity

TODO

  > tester IN sans RT
  
  > relancer mêmes tests avec + d'energy_absorption
  > relancer mêmes test avec RT 
  > relancer même tests sur OUT 

OBSERVATIONS :

  - Plus il y a de micros plus le temps de calcul des images sources et long

  Erreur FAUX Négatif ?
    l'Erreur suivante survient parfois, raytracing ou non, mais n'est pas persitente quand on relance l'exact même test
    LOG
      Traceback (most recent call last):
      File "C:\Users\Guillaume\pyroomacousticsGIP\CustomsScripts\loop.py", line 333, in <module>
        TheatreRoom.add_source(sourcePos, signal=anechoicAudioSource)
        ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py", line 2175, in add_source
        return self.add(SoundSource(position, signal=signal, delay=delay))
               ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "C:\Users\Guillaume\AppData\Local\Programs\Python\Python313\Lib\site-packages\pyroomacoustics\room.py", line 2001, in add
        raise ValueError("The source must be added inside the room.")
      ValueError: The source must be added inside the room.