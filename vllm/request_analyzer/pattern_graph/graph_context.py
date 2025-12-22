from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from vllm.logger import init_logger
from vllm.request_analyzer.pattern_graph.similarity import (
    Default_DeepResearch_Requests_Pattern, Default_DeepResearch_Stage,
    DynamicClustering, Graph, DeepResearchStructure, ToTStructure)

logger = init_logger(__name__)


class GraphMatchContext:
    def __init__(
        self,
        *,
        generation_tokenizer,
        use_all_node: bool,
        stage_ratio_method: str,
        graph_structure_type: str,
        graph_matching_mode: str,
        use_total_deadline: bool,
    ) -> None:
        self.generation_tokenizer = generation_tokenizer
        self.use_all_node = use_all_node
        self.stage_ratio_method = stage_ratio_method
        self.graph_structure_type = graph_structure_type
        self.graph_matching_mode = graph_matching_mode
        self.use_total_deadline = use_total_deadline

        self.dynamic_clustering = DynamicClustering()
        self.collection_graph_set: Set[Graph] = set()
        self.collection_graph_unfinished_dict: Dict[int, ToTStructure] = {}
        self.collection_deepresearch_unfinished_dict: Dict[int,
                                                          DeepResearchStructure] = {}
        self.deepresearch_base_graphs: List[Graph] = []

    def add_deepresearch_base_graphs(self, graphs: List[Graph]) -> None:
        self.deepresearch_base_graphs = graphs
        for graph in graphs:
            self.collection_graph_set.add(graph)

    def init_collection(self,
                        collection_id: int,
                        is_deepresearch: bool,
                        num_stages: int,
                        requests_per_stage: Optional[List[int]]) -> None:
        if not is_deepresearch:
            if collection_id not in self.collection_graph_unfinished_dict:
                self.collection_graph_unfinished_dict[
                    collection_id] = ToTStructure(
                        use_all_node=self.use_all_node,
                        stage_ratio_method=self.stage_ratio_method)
            tot_structure = self.collection_graph_unfinished_dict[collection_id]
            if tot_structure.is_finished:
                tot_structure.reset()
            return

        if collection_id not in self.collection_deepresearch_unfinished_dict:
            self.collection_deepresearch_unfinished_dict[
                collection_id] = DeepResearchStructure(
                    num_stages,
                    requests_per_stage,
                    self.use_all_node,
                    self.stage_ratio_method,
                )
        deepresearch_structure = self.collection_deepresearch_unfinished_dict[
            collection_id]
        if deepresearch_structure.is_finished:
            deepresearch_structure.reset()

    def calculate_stage_ratio(self,
                              num_stages: int,
                              requests_per_stage: Optional[List[int]],
                              prompt: str,
                              collection_id: int,
                              *,
                              is_deepresearch: bool = False,
                              stage_id: int = 0,
                              accumulate_stage_ratio: float = 0.0) -> float:
        if self.use_total_deadline:
            return 1.0

        default_ratio = self._default_ratio(is_deepresearch, stage_id)
        if self.graph_matching_mode == "none":
            return default_ratio
        if self.graph_matching_mode == "precise":
            if is_deepresearch and accumulate_stage_ratio > 0.0:
                logger.info(
                    f"Using precise accumulate_stage_ratio: {accumulate_stage_ratio}"
                )
                return accumulate_stage_ratio
            return default_ratio
        if self.graph_matching_mode == "static":
            return self._static_ratio(num_stages, requests_per_stage, prompt,
                                      collection_id, is_deepresearch, stage_id,
                                      default_ratio)
        if self.graph_matching_mode == "online":
            return self._online_ratio(num_stages, requests_per_stage, prompt,
                                      collection_id, is_deepresearch, stage_id)

        return default_ratio

    def track_collection_completion(self,
                                    collection_id: int,
                                    is_deepresearch: bool,
                                    request_input_length: int,
                                    request_output_length: int) -> None:
        if not is_deepresearch:
            tot_structure = self.collection_graph_unfinished_dict.get(
                collection_id)
            if tot_structure is None:
                return
            collection_finish = tot_structure.add_length(
                request_input_length, request_output_length)
            if collection_finish:
                if (tot_structure.stage_finish_time
                        and tot_structure.stage_in_out_lengths):
                    finished_graph = tot_structure.convert_to_graph()
                    self.collection_graph_set.add(finished_graph)
                    self.dynamic_clustering.add_finished_request(finished_graph)
                tot_structure.is_finished = True
            return

        deepresearch_structure = self.collection_deepresearch_unfinished_dict.get(
            collection_id)
        if deepresearch_structure is None:
            return
        collection_finish = deepresearch_structure.add_length(
            request_input_length, request_output_length)
        if collection_finish:
            if (deepresearch_structure.stage_in_out_lengths
                    and deepresearch_structure.stage_timings):
                finished_graph = deepresearch_structure.convert_to_graph()
                self.collection_graph_set.add(finished_graph)
                self.dynamic_clustering.add_finished_request(finished_graph)
            deepresearch_structure.is_finished = True

    def _default_ratio(self, is_deepresearch: bool, stage_id: int) -> float:
        if is_deepresearch:
            return 1.0 / Default_DeepResearch_Stage
        from vllm.request_analyzer.pattern_graph.similarity import Default_ToT_Requests_Pattern
        stage_id = min(stage_id, len(Default_ToT_Requests_Pattern) - 1)
        return Default_ToT_Requests_Pattern[stage_id] / sum(
            Default_ToT_Requests_Pattern)

    def _encode_prompt_len(self, prompt: str) -> int:
        return len(
            self.generation_tokenizer.encode(prompt, add_special_tokens=True))

    def _static_ratio(self, num_stages: int, requests_per_stage: Optional[List[
            int]], prompt: str, collection_id: int, is_deepresearch: bool,
                     stage_id: int, default_ratio: float) -> float:
        if stage_id != 0:
            return default_ratio
        if not is_deepresearch:
            tot_structure = self.collection_graph_unfinished_dict[collection_id]
            unfinished_graph = tot_structure.convert_to_unfinished_graph(
                self._encode_prompt_len(prompt), None)
            best_match = self.dynamic_clustering.find_best_match(
                unfinished_graph)
            stage = len(unfinished_graph.nodes)
            if best_match and best_match.times:
                if stage < len(best_match.times):
                    return best_match.times[stage] / sum(best_match.times)
                from vllm.request_analyzer.pattern_graph.similarity import Default_ToT_Requests_Pattern
                return Default_ToT_Requests_Pattern[stage] / sum(
                    Default_ToT_Requests_Pattern)
            from vllm.request_analyzer.pattern_graph.similarity import Default_ToT_Requests_Pattern
            return Default_ToT_Requests_Pattern[stage] / sum(
                Default_ToT_Requests_Pattern)

        deepresearch_structure = self.collection_deepresearch_unfinished_dict[
            collection_id]
        unfinished_graph = deepresearch_structure.convert_to_unfinished_graph(
            self._encode_prompt_len(prompt), None)
        best_match = self.dynamic_clustering.find_best_match(unfinished_graph)
        if best_match and best_match.times and stage_id < len(best_match.times):
            ratio_1 = best_match.times[stage_id] / sum(best_match.times)
            ratio_2 = Default_DeepResearch_Requests_Pattern[stage_id] / sum(
                Default_DeepResearch_Requests_Pattern)
            return max(ratio_1, ratio_2)
        return sum(Default_DeepResearch_Requests_Pattern[:stage_id + 1]) / sum(
            Default_DeepResearch_Requests_Pattern)

    def _online_ratio(self, num_stages: int, requests_per_stage: Optional[List[
            int]], prompt: str, collection_id: int, is_deepresearch: bool,
                      stage_id: int) -> float:
        if not is_deepresearch:
            tot_structure = self.collection_graph_unfinished_dict[collection_id]
            unfinished_graph = tot_structure.convert_to_unfinished_graph(
                self._encode_prompt_len(prompt), None)
            best_match = self.dynamic_clustering.find_best_match(
                unfinished_graph)
            stage = len(unfinished_graph.nodes) - 1
            logger.info(
                f"ToT collection_id: {collection_id}, stage: {stage}, best_match: {best_match}"
            )
            from vllm.request_analyzer.pattern_graph.similarity import Default_ToT_Requests_Pattern
            if best_match and best_match.times:
                if stage < len(best_match.times):
                    return sum(best_match.times[:stage]) / sum(best_match.times)
                return sum(Default_ToT_Requests_Pattern[:stage]) / sum(
                    Default_ToT_Requests_Pattern)
            return sum(Default_ToT_Requests_Pattern[:stage]) / sum(
                Default_ToT_Requests_Pattern)

        deepresearch_structure = self.collection_deepresearch_unfinished_dict[
            collection_id]
        unfinished_graph = deepresearch_structure.convert_to_unfinished_graph(
            self._encode_prompt_len(prompt), None)
        best_match = self.dynamic_clustering.find_best_match(unfinished_graph)
        logger.info(
            f"DeepResearch collection_id: {collection_id}, stage_id: {stage_id}, best_match: {best_match}"
        )
        if best_match and best_match.times and stage_id < len(best_match.times):
            ratio_1 = sum(best_match.times[:stage_id]) / sum(best_match.times)
            ratio_2 = sum(Default_DeepResearch_Requests_Pattern[:stage_id]) / sum(
                Default_DeepResearch_Requests_Pattern)
            return max(ratio_1, ratio_2)
        return sum(Default_DeepResearch_Requests_Pattern[:stage_id]) / sum(
            Default_DeepResearch_Requests_Pattern)
