#include <atomic>

#include "third_party/absl/base/call_once.h"

namespace {

absl::Status SetExecutorStructFn(
    void* library_handle) {  // TENSORFLOW_STATUS_OK
  static absl::once_flag flag;
  absl::call_once(flag, [library_handle]() {
    auto* executor_fn = stream_executor::tpu::ExecutorApiFn();
    TfTpu_ExecutorApiFn temp = {};

    TFTPU_SET_FN(&temp, TpuPlatform_New);
    TFTPU_SET_FN(&temp, TpuPlatform_Free);
    TFTPU_SET_FN(&temp, TpuPlatform_Initialize);
    TFTPU_SET_FN(&temp, TpuPlatform_Initialized);
    TFTPU_SET_FN(&temp, TpuPlatform_GetExecutor);
    TFTPU_SET_FN(&temp, TpuPlatform_Id);
    TFTPU_SET_FN(&temp, TpuPlatform_VisibleDeviceCount);
    TFTPU_SET_FN(&temp, TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopy);
    TFTPU_SET_FN(&temp, TpuPlatform_GetTopologyPtr);
    TFTPU_SET_FN(&temp, TpuPlatform_GetHostLocation);
    TFTPU_SET_FN(&temp, TpuPlatform_GetRuntimeVersion);

    TFTPU_SET_FN(&temp, TpuExecutor_Init);
    TFTPU_SET_FN(&temp, TpuExecutor_Free);
    TFTPU_SET_FN(&temp, TpuExecutor_Allocate);
    TFTPU_SET_FN(&temp, TpuExecutor_Deallocate);
    TFTPU_SET_FN(&temp, TpuExecutor_GetAllocatorStats);
    TFTPU_SET_FN(&temp, TpuExecutor_DeviceMemoryUsage);
    TFTPU_SET_FN(&temp, TpuExecutor_DeallocateStream);
    TFTPU_SET_FN(&temp, TpuExecutor_CreateStreamDependency);
    TFTPU_SET_FN(&temp, TpuExecutor_GetStatus);
    TFTPU_SET_FN(&temp, TpuExecutor_GetCoreLocation);
    TFTPU_SET_FN(&temp, TpuExecutor_AllocateEvent);
    TFTPU_SET_FN(&temp, TpuExecutor_RecordEvent);
    TFTPU_SET_FN(&temp, TpuExecutor_WaitForEvent);
    TFTPU_SET_FN(&temp, TpuExecutor_SynchronousMemcpyToHost);
    TFTPU_SET_FN(&temp, TpuExecutor_SynchronousMemcpyFromHost);
    TFTPU_SET_FN(&temp, TpuExecutor_MemcpyToHost);
    TFTPU_SET_FN(&temp, TpuExecutor_MemcpyFromHost);
    TFTPU_SET_FN(&temp, TpuExecutor_EnqueueInfeed);
    TFTPU_SET_FN(&temp, TpuExecutor_DequeueOutfeed);
    TFTPU_SET_FN(&temp, TpuExecutor_BlockHostUntilDone);
    TFTPU_SET_FN(&temp, TpuExecutor_SynchronizeAllActivity);
    TFTPU_SET_FN(&temp, TpuExecutor_UnloadAllPrograms);
    TFTPU_SET_FN(&temp, TpuExecutor_EnqueueCompactionOnStreamForHbm);

    TFTPU_SET_FN(&temp, TpuStream_New);
    TFTPU_SET_FN(&temp, TpuStream_Free);
    TFTPU_SET_FN(&temp, TpuStream_Stream);
    TFTPU_SET_FN(&temp, TpuStream_Status);
    TFTPU_SET_FN(&temp, TpuStream_IsSameSharedMemoryLocation);
    TFTPU_SET_FN(&temp, TpuStream_EnqueueTransferHostToDevice);
    TFTPU_SET_FN(&temp, TpuStream_EnqueueTransferDeviceToHost);
    TFTPU_SET_FN(&temp, TpuStream_TpuEnqueueOnDeviceSendRecvLocal);

    TFTPU_SET_FN(&temp, TpuEvent_New);
    TFTPU_SET_FN(&temp, TpuEvent_Free);

    TFTPU_SET_FN(&temp, TpuStatus_New);
    TFTPU_SET_FN(&temp, TpuStatus_Create);
    TFTPU_SET_FN(&temp, TpuStatus_Set);
    TFTPU_SET_FN(&temp, TpuStatus_Free);
    TFTPU_SET_FN(&temp, TpuStatus_Message);
    TFTPU_SET_FN(&temp, TpuStatus_Code);
    TFTPU_SET_FN(&temp, TpuStatus_Ok);

    TFTPU_SET_FN(&temp, TpuDeviceDescription_New);
    TFTPU_SET_FN(&temp, TpuDeviceDescription_Free);

    TFTPU_SET_FN(&temp, TpuExecutor_CreateDeviceDescription);
    TFTPU_SET_FN(&temp, TpuExecutor_HostCallback);

    TFTPU_SET_FN(&temp, TpuTransferManager_New);
    TFTPU_SET_FN(&temp, TpuTransferManager_Free);
    TFTPU_SET_FN(&temp, TpuTransferManager_PlatformId);
    TFTPU_SET_FN(&temp, TpuTransferManager_HostShapeToDeviceShape);
    TFTPU_SET_FN(&temp, TpuTransferManager_TransferLiteralToDeviceAsync);
    TFTPU_SET_FN(&temp, TpuTransferManager_TransferLiteralFromDevice);
    TFTPU_SET_FN(&temp, TpuTransferManager_GetByteSizeRequirement);
    TFTPU_SET_FN(&temp, TpuTransferManager_ChooseCompactLayoutForShape);
    TFTPU_SET_FN(&temp, TpuTransferManager_CanShapedBufferBeAccessedNow);
    TFTPU_SET_FN(&temp, TpuTransferManager_CanBufferBeAccessedNow);
    TFTPU_SET_FN(&temp, TpuTransferManager_WriteSingleTupleIndexTable);
    TFTPU_SET_FN(&temp, TpuTransferManager_GetInfeedLayout);
    TFTPU_SET_FN(&temp, TpuTransferManager_LinearizeToBuffers);
    TFTPU_SET_FN(&temp, TpuTransferManager_FreeBuffers);
    TFTPU_SET_FN(&temp, TpuTransferManager_TransferLiteralToInfeed);
    TFTPU_SET_FN(&temp, TpuTransferManager_TransferBuffersToInfeed);
    TFTPU_SET_FN(&temp, TpuTransferManager_TransferLiteralFromOutfeed);
    TFTPU_SET_FN(&temp, TpuTransferManager_ResetDevices);
    TFTPU_SET_FN(&temp, TpuTransferManager_ReadDynamicShapes);

    TFTPU_SET_FN(&temp, TpuComputationPlacer_New);
    TFTPU_SET_FN(&temp, TpuComputationPlacer_Free);
    TFTPU_SET_FN(&temp, TpuComputationPlacer_AssignDevices);
    TFTPU_SET_FN(&temp, TpuComputationPlacer_AssignLocalDevices);

    TFTPU_SET_FN(&temp, TpuTopology_LogicalDevicesPerHost);
    TFTPU_SET_FN(&temp, TpuTopology_LogicalDevicesPerChip);
    TFTPU_SET_FN(&temp, TpuTopology_HostCount);
    TFTPU_SET_FN(&temp, TpuTopology_ChipsPerHost);
    TFTPU_SET_FN(&temp, TpuTopology_ChipBounds_X);
    TFTPU_SET_FN(&temp, TpuTopology_ChipBounds_Y);
    TFTPU_SET_FN(&temp, TpuTopology_ChipBounds_Z);
    TFTPU_SET_FN(&temp, TpuTopology_HasChip);
    TFTPU_SET_FN(&temp, TpuTopology_CoreForId);
    TFTPU_SET_FN(&temp, TpuTopology_Core);
    TFTPU_SET_FN(&temp, TpuTopology_NumCores);
    TFTPU_SET_FN(&temp, TpuTopology_Cores);
    TFTPU_SET_FN(&temp, TpuTopology_IdForHost);
    TFTPU_SET_FN(&temp, TpuTopology_Version);

    TFTPU_SET_FN(&temp, TpuCoreLocation_ChipCoordinates);
    TFTPU_SET_FN(&temp, TpuCoreLocation_HostCoordinates);
    TFTPU_SET_FN(&temp, TpuCoreLocation_Index);
    TFTPU_SET_FN(&temp, TpuCoreLocation_Id);

    TFTPU_SET_FN(&temp, TpuHostLocation_Id);
    TFTPU_SET_FN(&temp, TpuHostLocation_NumCores);
    TFTPU_SET_FN(&temp, TpuHostLocation_Cores);

    TFTPU_SET_FN(&temp, TpuCompiler_New);
    TFTPU_SET_FN(&temp, TpuCompiler_Free);

    TFTPU_SET_FN(&temp, TpuCompiler_RunHloPasses);
    TFTPU_SET_FN(&temp, TpuCompiler_RunBackend);
    TFTPU_SET_FN(&temp, TpuCompiler_Compile);
    TFTPU_SET_FN(&temp, TpuCompiler_ShapeSize);
    TFTPU_SET_FN(&temp, TpuCompiler_DefaultDeviceShapeRepresentation);
    TFTPU_SET_FN(&temp, TpuExecutable_ExecuteAsyncOnStream);
    TFTPU_SET_FN(&temp, TpuExecutable_FreeXlaShapeIndexArray);
    TFTPU_SET_FN(&temp, TpuExecutable_FreeMaybeOwningDeviceAddressArray);
    TFTPU_SET_FN(&temp, TpuExecutable_Fingerprint);
    TFTPU_SET_FN(&temp, TpuExecutable_Serialize);
    TFTPU_SET_FN(&temp, TpuExecutableSerialize_GetByteSize);
    TFTPU_SET_FN(&temp, TpuExecutableSerialize_WriteToArray);
    TFTPU_SET_FN(&temp, TpuExecutableSerialize_FreeHandle);
    TFTPU_SET_FN(&temp, TpuExecutable_Deserialize);
    TFTPU_SET_FN(&temp, TpuExecutable_HloModule);
    TFTPU_SET_FN(&temp, TpuExecutable_Free);

    TFTPU_SET_FN(&temp, XlaShapeToTpuShapeRepresentation);
    TFTPU_SET_FN(&temp, XlaShapeToTpuPaddedShape);

    TFTPU_SET_FN(&temp, TpuAsyncCollectiveOffloadHelper_Init);
    TFTPU_SET_FN(&temp, TpuAsyncCollectiveOffloadHelper_Shutdown);

    void* platform_new_fn = reinterpret_cast<void*>(temp.TpuPlatform_NewFn);
    temp.TpuPlatform_NewFn = nullptr;
    *executor_fn = temp;
    std::atomic_thread_fence(std::memory_order_release);
    reinterpret_cast<std::atomic<void*>*>(&executor_fn->TpuPlatform_NewFn)
        ->store(platform_new_fn, std::memory_order_release);
  });

  return absl::OkStatus();
}

}  // namespace
