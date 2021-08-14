import numpy as np
import os

def dataset_gen(dataset_path, cityname, cate_num,
          poi_norm,
          flow_norm,
          checkin_norm, seed=0):
    
    POI_inside_path = os.path.join(datasetPath, '{0}_poi_{1}_inside.npy'.format(cityname, cate_num))
    POI_outside_path = os.path.join(datasetPath, '{0}_poi_{1}_outside.npy'.format(cityname, cate_num))
    inflow_path = os.path.join(datasetPath, '{0}_inflow.npy'.format(cityname))
    outflow_path = os.path.join(datasetPath, '{0}_outflow.npy'.format(cityname))
    checkin_path = os.path.join(datasetPath, '{0}_checkin_inside.npy'.format(cityname))
    
    poi_in = np.load(os.path.join(datasetPath, POI_inside_path))[1:].astype(np.float32)
    poi_out = np.load(os.path.join(datasetPath, POI_outside_path))[1:].astype(np.float32)
    poi = np.c_[poi_in, poi_out]
            
    # Drop the first region because it's the region of whole city.
    inflow_data = np.load(os.path.join(dataset_path, inflow_path))[1:, :].astype(np.float32) # [region, time]
    outflow_data = np.load(os.path.join(dataset_path, outflow_path))[1:, :].astype(np.float32) # [region, time]
    checkin_data = np.load(os.path.join(dataset_path, checkin_path))[1:, :].astype(np.float32) # [region, time, cate]
    
    space_slice = np.shape(inflow_data)[0]
    time_slice = 48
    recordNum = int(space_slice * time_slice)
    
    if cityname is 'shanghai':
        days_list = [0,3,4,5,6]
    if cityname is 'beijing':
        days_list = [1,2,3,4,5, 8,9,10,11,12, 15,16,17,18,19, 22,23,24,25,26]
    for day in days_list:
        inflow += inflowData[:, timeSlice*day:timeSlice*(day+1)]
        outflow += outflowDatainflowData[:, timeSlice*day:timeSlice*(day+1)]
        checkin += checkinData[:, timeSlice*day:timeSlice*(day+1), :]
    inflow /= len(days_list)
    outflow /= len(days_list)
    checkin /= len(days_list)

    if flow_norm:
        inflow_max = np.max(inflow)
        outflow_max = np.max(outflow)
        # Normalized the inflow into [0, 1]
        inflow = inflow / inflow_max
        outflow = outflow / outflow_max
    
    # TF-IDF for check-in
    if checkin_norm:
        for iter in range(14):
            checkin[:, :, iter] /= np.max(checkin[:, :, iter])
            
    order = np.random.permutation(np.arange(recordNum))
    time = np.floor(order / spaceSlice).astype(int)
    loc = np.mod(order, spaceSlice).astype(int)
    
    poiRecord_o = poi[loc, :]
    inflowTarget = np.reshape(inflow[loc, time], [recordNum, 1])
    outflowTarget = np.reshape(outflow[loc, time], [recordNum, 1])
    checkinTarget = checkin[loc, time, :]
    timeInOneDay = np.reshape(np.mod(time, timeSlice), [recordNum, 1])
    
    poiRecord = poiRecord_o
    
    np.save(os.path.join(datasetPath, 'record', 'inflowgt.npy'), inflow)
    np.save(os.path.join(datasetPath, 'record', 'outflowgt.npy'), outflow)
    np.save(os.path.join(datasetPath, 'record', 'loc.npy'), loc)
    np.save(os.path.join(datasetPath, 'record', 'time.npy'), time)
    np.save(os.path.join(datasetPath, 'record', 'input1.npy'), timeInOneDay)
    np.save(os.path.join(datasetPath, 'record', 'input2.npy'), poiRecord)
    np.save(os.path.join(datasetPath, 'record', 'target1i.npy'), inflowTarget)
    np.save(os.path.join(datasetPath, 'record', 'target1o.npy'), outflowTarget)
    np.save(os.path.join(datasetPath, 'record', 'target2.npy'), checkinTarget)
    print('Gen dataset in ' + os.path.join(datasetPath, 'record'))
    
    

def dataRead(datasetPath, flow):
    assert(flow is 'i' or flow is 'o')
    
    
    loc = np.load(os.path.join(datasetPath, 'record', 'loc.npy'))
    time = np.load(os.path.join(datasetPath, 'record', 'time.npy'))
    input1 = np.load(os.path.join(datasetPath, 'record', 'input1.npy'))
    input2 = np.load(os.path.join(datasetPath, 'record', 'input2.npy'))
    if flow is 'i':
        target1 = np.load(os.path.join(datasetPath, 'record', 'target1i.npy'))
    else:
        target1 = np.load(os.path.join(datasetPath, 'record', 'target1o.npy'))
    target2 = np.load(os.path.join(datasetPath, 'record', 'target2.npy'))
    
    recordNum = np.shape(loc)[0]
    
    trainSize = 0.7
    validSize = 0.15
    testSize = 0.15
    
    trainLen = int(recordNum*trainSize)
    validLen = int(recordNum*validSize)
    testLen = recordNum - trainLen - validLen
    
    # Divide the dataset into trainset and testset.
    trainList = np.arange(recordNum)[:trainLen]
    validList = np.arange(recordNum)[trainLen:(trainLen + validLen)]
    testList = np.arange(recordNum)[(trainLen + validLen):]
    
    # Define the input and output of the model
    trainInput1 = input1[trainList]
    trainInput2 = input2[trainList]
    trainTarget1 = target1[trainList]
    trainTarget2 = target2[trainList]
    
    validInput1 = input1[validList]
    validInput2 = input2[validList]
    validTarget1 = target1[validList]
    validTarget2 = target2[validList]
    
    testInput1 = input1[testList]
    testInput2 = input2[testList]
    testTarget1 = target1[testList]
    testTarget2 = target2[testList]
    
    return trainInput1, trainInput2, trainTarget1, trainTarget2, trainLen,\
         validInput1, validInput2, validTarget1, validTarget2, validList,\
         testInput1, testInput2, testTarget1, testTarget2, testLen,\
        loc, time


def inflowMax(datasetPath, inflowPath):
    inflowData = np.load(os.path.join(datasetPath, inflowPath))[1:, :].astype(np.float32)
    spaceSlice = np.shape(inflowData)[0]
    timeSlice = 48
    inflow = (inflowData[:, 192:240] + inflowData[:, 240:288] + inflowData[:, 288:336]) / 3
    return np.max(inflow)


def outflowMax(datasetPath, outflowPath):
    outflowData = np.load(os.path.join(datasetPath, outflowPath))[1:, :].astype(np.float32)
    spaceSlice = np.shape(outflowData)[0]
    timeSlice = 48
    outflow = (outflowData[:, 192:240] + outflowData[:, 240:288] + outflowData[:, 288:336]) / 3
    return np.max(outflow)