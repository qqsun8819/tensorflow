/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Contains utilities for clustering compilable graph nodes via XLA.

#ifndef TENSORFLOW_COMPILER_JIT_MLIR_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_MLIR_UTIL_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/mlir/tf_mlir_compiler.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

class MlirExecutableClosure {
 public:
  explicit MlirExecutableClosure(
      const std::string& graphstr,
      const std::string& entry_func_name) {
    mlir_compiler_ = new SimpleMlirCompiler(graphstr, entry_func_name);
    mlir_compiler_->CompileGraphDef(true);
  }

  SimpleMlirCompiler* compiler() const { return mlir_compiler_;} 

 private:
  SimpleMlirCompiler* mlir_compiler_;
  
  TF_DISALLOW_COPY_AND_ASSIGN(MlirExecutableClosure);
};

class MlirExecutableClosureStore {
 public:
  MlirExecutableClosureStore()  {}

  using KeyT = string;

  KeyT Produce(const KeyT& key,
               const std::string& graph_def_string,
               const std::string& entry_func_name) {
    mutex_lock l(mutex_);
    MlirExecutableClosure* result =
        new MlirExecutableClosure(graph_def_string, entry_func_name);
    bool insert_successful = closures_.emplace(key, result).second;
    DCHECK(insert_successful);
    (void)insert_successful;
    return key;
  }

  KeyT Produce(const KeyT& key,
               const GraphDef& graph_def,
               const std::string& entry_func_name) {
    return Produce(key, graph_def.DebugString(), entry_func_name);
  }

  MlirExecutableClosure* Consume(const KeyT& key) {
    // TODO: use read-write lock here
    mutex_lock l(mutex_);
    auto it = closures_.find(key);
    DCHECK(it != closures_.end());
    return it->second;
  }

  static MlirExecutableClosureStore* Global() {
    static MlirExecutableClosureStore* instance = new MlirExecutableClosureStore;
    return instance;
  }

 private:
  mutex mutex_;
  absl::flat_hash_map<KeyT, MlirExecutableClosure*> closures_ GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(MlirExecutableClosureStore);
};

class MlirSubGraphDefStore {
 public:
  MlirSubGraphDefStore() {}

  void StoreSubGraph(const std::string& key, const std::string& graph_def_string) {
    mutex_lock l(mutex_);
    cache_[key] = graph_def_string;
  }

  void StoreSubGraph(const std::string& key, const GraphDef& graph_def) {
    mutex_lock l(mutex_);
    cache_[key] = graph_def.DebugString();;
  }

  void StoreSubGraph(const std::string& key, Graph* graph) {
    mutex_lock l(mutex_);
    GraphDef graph_def;
    graph->ToGraphDef(&graph_def);
    cache_[key] = graph_def.DebugString();
  }

  std::string GetSubGraphDefString(const std::string& key) {
    mutex_lock l(mutex_);
    auto it = cache_.find(key);
    DCHECK(it != cache_.end());
    return it->second;
  }

  static MlirSubGraphDefStore* Global() {
    static MlirSubGraphDefStore* instance = new MlirSubGraphDefStore;
    return instance;
  }

 private:
  mutex mutex_;
  absl::flat_hash_map<std::string, std::string> cache_ GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(MlirSubGraphDefStore);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_MLIR_UTIL_H_
