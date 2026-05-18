/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/hlo_module_config.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/service/computation_layout.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/schedule_config.h"
#include "xla/service/sharding_config.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {

using absl::StrAppend;

HloModuleConfig::HloModuleConfig(const ProgramShape& program_shape,
                                 bool ignore_layouts)
    : entry_computation_layout_(
          ComputationLayout(program_shape, ignore_layouts)) {}

HloModuleConfig::HloModuleConfig(ComputationLayout entry_computation_layout)
    : entry_computation_layout_(std::move(entry_computation_layout)) {}

HloModuleConfig HloModuleConfig::Share() {
  std::shared_ptr<Shareable> shared = std::visit(
      [&](auto&& s) {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, Shareable>) {
          return std::make_shared<Shareable>(std::forward<decltype(s)>(s));
        } else {
          return s;
        }
      },
      std::move(shareable_));
  shareable_ = shared;

  HloModuleConfig other;
  other.shareable_ = std::move(shared);
  return other;
}

void HloModuleConfig::SetDefaultComputationLayout(
    const ProgramShape& program_shape) {
  entry_computation_layout_ = ComputationLayout(program_shape);
}

void HloModuleConfig::SetComputationLayoutIfExists(
    const ProgramShape& program_shape) {
  entry_computation_layout_ = ComputationLayout(program_shape,
                                                /*ignore_layouts=*/false);
}

std::string HloModuleConfig::compilation_cache_key() const {
  std::string key = absl::StrCat("profiling=", hlo_profiling_enabled());
  StrAppend(&key, "::(");
  std::vector<std::string> params;
  if (entry_computation_layout_.has_value()) {
    for (const ShapeLayout& param_layout :
         entry_computation_layout_->parameter_layouts()) {
      params.push_back(param_layout.shape().ToString());
    }
    StrAppend(&key, absl::StrJoin(params, ", "), ") => ",
              entry_computation_layout_->result_shape()
                  .ToProto()
                  .SerializeAsString());
  }
  if (seed() != 0) {
    static std::atomic<int> counter{0};
    StrAppend(&key, "forcing recompile ", counter++);
  }
  StrAppend(&key, "::exec_time_optimization_effort=",
            exec_time_optimization_effort());
  StrAppend(&key, "::memory_fitting_effort=", memory_fitting_effort());
  StrAppend(&key, "::optimization_level=", optimization_level());
  StrAppend(&key, "::memory_fitting_level=", memory_fitting_level());
  if (replica_count() != 1) {
    StrAppend(&key, "::replica_count=", replica_count());
  }
  StrAppend(&key, debug_options().DebugString());
  if (intra_op_parallelism_threads() > 0) {
    StrAppend(&key, "::intra_op_parallelism_threads=",
              intra_op_parallelism_threads());
  }
  if (!device_type().empty()) {
    StrAppend(&key, device_type());
  }
  StrAppend(&key, "::alias_passthrough_params=", alias_passthrough_params());
  StrAppend(&key, "::allow_spmd_sharding_propagation_to_parameters={",
            absl::StrJoin(allow_spmd_sharding_propagation_to_parameters(), ","),
            "}");
  StrAppend(&key, "::allow_spmd_sharding_propagation_to_output={",
            absl::StrJoin(allow_spmd_sharding_propagation_to_output(), ","),
            "}");
  if (!fdo_profile().empty()) {
    StrAppend(&key, "::fdo_profile=", absl::BytesToHexString(fdo_profile()));
  }
  if (device_memory_size() != 0) {
    StrAppend(&key, "::device_address_size=", device_memory_size());
  }
  StrAppend(&key, "::use_shardy_partitioner=", use_shardy_partitioner());
  if (partition_size() != 0) {
    StrAppend(&key, "::partition_size=", partition_size());
  }
  return key;
}

/*static*/ void HloModuleConfig::AssignProtoShardableValueUpdatePairs(
    tsl::protobuf::RepeatedPtrField<ShardableValueUpdatePairProto>*
        proto_update_pairs,
    const std::vector<HloModuleConfig::ShardableValueUpdatePair>&
        update_pairs) {
  using ProtoShard = std::decay_t<decltype(proto_update_pairs->at(0))>;
  proto_update_pairs->Reserve(update_pairs.size());

  for (const auto& pair : update_pairs) {
    ProtoShard shard;
    shard.set_input_parameter_number(pair.input_parameter_number);
    for (int64_t val : pair.parameter_shape_index) {
      shard.add_parameter_shape_index(val);
    }
    for (int64_t val : pair.output_shape_index) {
      shard.add_output_shape_index(val);
    }
    proto_update_pairs->Add(std::move(shard));
  }
}

static HloModuleConfigProto::BoolList BoolVectorToProto(
    const std::vector<bool>& vals) {
  HloModuleConfigProto::BoolList list;
  for (int i = 0; i < vals.size(); ++i) {
    list.add_vals(vals[i]);
  }
  return list;
}

static void AssignProtoFusionConfig(
    HloModuleConfigProto& proto,
    const std::vector<std::vector<bool>>& fusion_config) {
  auto* proto_config = proto.mutable_fusion_config();
  proto_config->Reserve(fusion_config.size());
  for (const auto& vals : fusion_config) {
    proto_config->Add(BoolVectorToProto(vals));
  }
}

static void AssignProtoDotConfig(
    HloModuleConfigProto& proto,
    const absl::flat_hash_map<std::string, std::vector<int64_t>>& dot_config) {
  std::map<std::string, std::vector<int64_t>> sorted_dot_config;
  sorted_dot_config.insert(dot_config.begin(), dot_config.end());
  for (const auto& [key, list_vector] : sorted_dot_config) {
    HloModuleConfigProto::Int64List list;
    for (int64_t val : list_vector) {
      list.add_vals(val);
    }
    proto.mutable_dot_config()->try_emplace(key, std::move(list));
  }
}

static void AssignProtoLayoutConfig(
    HloModuleConfigProto& proto,
    const std::vector<std::vector<std::vector<int64_t>>>& layout_config) {
  auto* proto_layout_config = proto.mutable_layout_config();
  proto_layout_config->Reserve(layout_config.size());
  for (const auto& config_row : layout_config) {
    HloModuleConfigProto::Int64ListList proto_list_list;
    proto_list_list.mutable_lists()->Reserve(config_row.size());
    for (const auto& cell : config_row) {
      HloModuleConfigProto::Int64List list;
      for (int64_t val : cell) {
        list.add_vals(val);
      }
      *proto_list_list.add_lists() = std::move(list);
    }
    proto_layout_config->Add(std::move(proto_list_list));
  }
}

static void AssignProtoPhaseOrderingConfig(
    HloModuleConfigProto& proto,
    const std::vector<std::vector<bool>>& phase_config) {
  auto* proto_config = proto.mutable_phase_ordering_config();
  proto_config->Reserve(phase_config.size());
  for (const auto& vals : phase_config) {
    proto_config->Add(BoolVectorToProto(vals));
  }
}

/*static*/ void HloModuleConfig::AssignStructShardableValueUpdatePairs(
    HloModuleConfig& config,
    const tsl::protobuf::RepeatedPtrField<ShardableValueUpdatePairProto>&
        pairs) {
  std::vector<HloModuleConfig::ShardableValueUpdatePair> cfg_pairs;
  cfg_pairs.reserve(pairs.size());
  for (const auto& proto_pair : pairs) {
    HloModuleConfig::ShardableValueUpdatePair pair;
    pair.input_parameter_number = proto_pair.input_parameter_number();
    const auto param_idx = proto_pair.parameter_shape_index();
    pair.parameter_shape_index.assign(param_idx.begin(), param_idx.end());
    const auto output_idx = proto_pair.output_shape_index();
    pair.output_shape_index.assign(output_idx.begin(), output_idx.end());
    cfg_pairs.push_back(pair);
  }
  config.set_shardable_value_update_pairs(std::move(cfg_pairs));
}

static void AssignStructFusionConfig(HloModuleConfig& config,
                                     const HloModuleConfigProto& proto) {
  std::vector<std::vector<bool>> module_config;
  auto& proto_config = proto.fusion_config();
  module_config.reserve(proto_config.size());
  for (auto& list : proto_config) {
    std::vector<bool> temp;
    for (bool val : list.vals()) {
      temp.push_back(val);
    }
    module_config.push_back(std::move(temp));
  }
  config.set_fusion_config(std::move(module_config));
}

static void AssignStructDotConfig(HloModuleConfig& config,
                                  const HloModuleConfigProto& proto) {
  auto& proto_config = proto.dot_config();
  for (auto& [key, int_list] : proto_config) {
    std::vector<int64_t> value{int_list.vals().begin(), int_list.vals().end()};
    config.mutable_dot_config()->insert(std::pair{key, value});
  }
}

static void AssignStructLayoutConfig(HloModuleConfig& config,
                                     const HloModuleConfigProto& proto) {
  std::vector<std::vector<std::vector<int64_t>>> module_config;
  auto proto_config = proto.layout_config();
  module_config.reserve(proto_config.size());
  for (const auto& proto_row_wrapper : proto_config) {
    const auto& proto_row = proto_row_wrapper.lists();
    std::vector<std::vector<int64_t>> module_row;
    module_row.reserve(proto_row.size());
    for (const auto& proto_cell : proto_row) {
      const auto& cell = proto_cell.vals();
      module_row.push_back(std::vector<int64_t>(cell.begin(), cell.end()));
    }
    module_config.push_back(std::move(module_row));
  }
  *config.mutable_layout_config() = std::move(module_config);
}

static void AssignStructPhaseOrderingConfig(HloModuleConfig& config,
                                            const HloModuleConfigProto& proto) {
  std::vector<std::vector<bool>> module_config;
  auto& proto_config = proto.phase_ordering_config();
  module_config.reserve(proto_config.size());
  for (auto& list : proto_config) {
    std::vector<bool> temp;
    for (bool val : list.vals()) {
      temp.push_back(val);
    }
    module_config.push_back(std::move(temp));
  }
  config.set_phase_ordering_config(std::move(module_config));
}

HloModuleConfigProto HloModuleConfig::ToProto() const {
  HloModuleConfigProto proto;
  if (has_entry_computation_layout()) {
    *proto.mutable_entry_computation_layout() =
        entry_computation_layout().ComputeProgramShape().ToProto();
  }
  proto.set_seed(seed());
  proto.set_launch_id(launch_id());
  proto.set_replica_count(replica_count());
  proto.set_num_partitions(num_partitions());
  for (bool requirement : param_requires_broadcast_via_collectives()) {
    proto.add_param_requires_broadcast_via_collectives(requirement);
  }
  proto.set_use_spmd_partitioning(use_spmd_partitioning());
  proto.set_use_auto_spmd_partitioning(use_auto_spmd_partitioning());
  for (int64_t partitioning_shape : auto_spmd_partitioning_mesh_shape()) {
    proto.add_auto_spmd_partitioning_mesh_shape(partitioning_shape);
  }
  for (int64_t partitioning_id : auto_spmd_partitioning_mesh_ids()) {
    proto.add_auto_spmd_partitioning_mesh_ids(partitioning_id);
  }
  proto.set_exec_time_optimization_effort(exec_time_optimization_effort());
  proto.set_memory_fitting_effort(memory_fitting_effort());
  proto.set_optimization_level(optimization_level());
  proto.set_memory_fitting_level(memory_fitting_level());
  proto.set_deduplicate_hlo(deduplicate_hlo());
  proto.set_intra_op_parallelism_threads(intra_op_parallelism_threads());
  proto.set_device_type(device_type());
  *proto.mutable_debug_options() = debug_options();

  if (has_static_device_assignment()) {
    auto proto_assignment = proto.mutable_static_device_assignment();
    static_device_assignment().Serialize(proto_assignment);
  }
  AssignProtoShardableValueUpdatePairs(
      proto.mutable_shardable_value_update_pairs(),
      shardable_value_update_pairs());
  proto.set_alias_passthrough_params(alias_passthrough_params());
  proto.set_content_aware_computation_sorting(
      content_aware_computation_sorting());
  proto.set_fusion_config_collection(
      static_cast<HloModuleConfigProto::FusionConfigCollection>(
          fusion_config_collection()));
  AssignProtoFusionConfig(proto, fusion_config());
  AssignProtoDotConfig(proto, dot_config());
  AssignProtoLayoutConfig(proto, layout_config());
  for (uint64_t cfg : memory_space_assignment_config()) {
    proto.add_memory_space_assignment_config(cfg);
  }
  AssignProtoPhaseOrderingConfig(proto, phase_ordering_config());
  proto.set_phase_index(phase_index());

  for (bool value : allow_spmd_sharding_propagation_to_parameters()) {
    proto.add_allow_spmd_sharding_propagation_to_parameters(value);
  }
  for (bool value : allow_spmd_sharding_propagation_to_output()) {
    proto.add_allow_spmd_sharding_propagation_to_output(value);
  }

  auto proto_analysis_map = proto.mutable_analysis_allowance_map();
  for (const auto& [key, value] : shareable().analysis_allowance_map) {
    proto_analysis_map->insert({std::string(key), value});
  }
  proto.set_matrix_unit_operand_precision(matrix_unit_operand_precision());
  proto.set_allow_separate_sharding_programs(
      allow_separate_sharding_programs());
  proto.set_fdo_profile(fdo_profile());
  proto.set_device_memory_size(device_memory_size());
  proto.set_use_shardy_partitioner(use_shardy_partitioner());
  proto.set_partition_size(partition_size());
  *proto.mutable_sharding_config() = ShardingConfig::ToProto(sharding_config());
  *proto.mutable_schedule_config() = ScheduleConfig::ToProto(schedule_config());
  return proto;
}

absl::StatusOr<std::unique_ptr<HloModuleConfig>>
HloModuleConfig::CreateFromProto(const HloModuleConfigProto& proto) {
  auto config = std::make_unique<HloModuleConfig>();

  if (proto.has_entry_computation_layout()) {
    TF_ASSIGN_OR_RETURN(
        auto comp_layout,
        ProgramShape::FromProto(proto.entry_computation_layout()));
    config->SetComputationLayoutIfExists(comp_layout);
  } else {
    config->clear_entry_computation_layout();
  }
  config->mutable_shareable().seed = proto.seed();
  config->mutable_shareable().launch_id = proto.launch_id();
  config->mutable_shareable().replica_count = proto.replica_count();
  config->mutable_shareable().num_partitions = proto.num_partitions();
  config->mutable_shareable().param_requires_broadcast_via_collectives.assign(
      proto.param_requires_broadcast_via_collectives().begin(),
      proto.param_requires_broadcast_via_collectives().end());
  config->mutable_shareable().use_spmd_partitioning =
      proto.use_spmd_partitioning();
  config->mutable_shareable().use_auto_spmd_partitioning =
      proto.use_auto_spmd_partitioning();
  config->mutable_shareable().auto_spmd_partitioning_mesh_shape.assign(
      proto.auto_spmd_partitioning_mesh_shape().begin(),
      proto.auto_spmd_partitioning_mesh_shape().end());
  config->mutable_shareable().auto_spmd_partitioning_mesh_ids.assign(
      proto.auto_spmd_partitioning_mesh_ids().begin(),
      proto.auto_spmd_partitioning_mesh_ids().end());
  config->mutable_shareable().exec_time_optimization_effort =
      proto.exec_time_optimization_effort();
  config->mutable_shareable().memory_fitting_effort =
      proto.memory_fitting_effort();
  config->mutable_shareable().optimization_level = proto.optimization_level();
  config->mutable_shareable().memory_fitting_level =
      proto.memory_fitting_level();
  config->mutable_shareable().deduplicate_hlo = proto.deduplicate_hlo();
  config->mutable_shareable().intra_op_parallelism_threads =
      proto.intra_op_parallelism_threads();
  config->mutable_shareable().device_type = proto.device_type();
  if (proto.has_debug_options()) {
    config->mutable_shareable().debug_options = proto.debug_options();
  }
  if (proto.has_static_device_assignment()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<DeviceAssignment> device_assignment,
        DeviceAssignment::Deserialize(proto.static_device_assignment()));
    config->mutable_shareable().static_device_assignment =
        std::move(*device_assignment);
  }
  AssignStructShardableValueUpdatePairs(*config,
                                        proto.shardable_value_update_pairs());
  config->mutable_shareable().alias_passthrough_params =
      proto.alias_passthrough_params();
  config->mutable_shareable().content_aware_computation_sorting =
      proto.content_aware_computation_sorting();
  config->mutable_shareable().fusion_config_collection =
      static_cast<FusionConfigCollection>(proto.fusion_config_collection());
  AssignStructFusionConfig(*config, proto);
  AssignStructDotConfig(*config, proto);
  AssignStructLayoutConfig(*config, proto);
  config->mutable_shareable().memory_space_assignment_config.assign(
      proto.memory_space_assignment_config().begin(),
      proto.memory_space_assignment_config().end());
  AssignStructPhaseOrderingConfig(*config, proto);
  config->mutable_shareable().phase_index = proto.phase_index();
  config->mutable_shareable()
      .allow_spmd_sharding_propagation_to_parameters.assign(
          proto.allow_spmd_sharding_propagation_to_parameters().begin(),
          proto.allow_spmd_sharding_propagation_to_parameters().end());
  config->mutable_shareable().allow_spmd_sharding_propagation_to_output.assign(
      proto.allow_spmd_sharding_propagation_to_output().begin(),
      proto.allow_spmd_sharding_propagation_to_output().end());
  config->mutable_shareable().analysis_allowance_map.insert(
      proto.analysis_allowance_map().begin(),
      proto.analysis_allowance_map().end());
  config->mutable_shareable().matrix_unit_operand_precision =
      proto.matrix_unit_operand_precision();
  config->mutable_shareable().allow_separate_sharding_programs =
      proto.allow_separate_sharding_programs();
  config->mutable_shareable().fdo_profile = proto.fdo_profile();
  config->mutable_shareable().device_memory_size = proto.device_memory_size();
  config->mutable_shareable().use_shardy_partitioner =
      proto.use_shardy_partitioner();
  config->mutable_shareable().partition_size = proto.partition_size();
  config->mutable_shareable().sharding_config =
      ShardingConfig::FromProto(proto.sharding_config());
  config->mutable_shareable().schedule_config =
      ScheduleConfig::FromProto(proto.schedule_config());
  return std::move(config);
}

HloModuleConfig::Shareable& HloModuleConfig::mutable_shareable() {
  return std::visit(
      [](auto&& s) -> Shareable& {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, Shareable>) {
          return s;
        } else {
          return *s;
        }
      },
      shareable_);
}

const HloModuleConfig::Shareable& HloModuleConfig::shareable() const {
  return std::visit(
      [](auto&& s) -> const Shareable& {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, Shareable>) {
          return s;
        } else {
          return *s;
        }
      },
      shareable_);
}

}  // namespace xla
