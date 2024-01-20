/**
 * This file is part of FLaME.
 * Copyright (C) 2017 W. Nicholas Greene (wng@csail.mit.edu)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see <http://www.gnu.org/licenses/>.
 *
 * @file nltgv2_l1_graph_regularizer.cc
 * @author W. Nicholas Greene
 * @date 2016-10-06 20:43:34 (Thu)
 */

#include "flame/optimizers/nltgv2_l1_graph_regularizer.h"

#include <limits>

namespace flame {

    namespace optimizers {

        namespace nltgv2_l1_graph_regularizer {

            void step(const Params &params, Graph *graph) {
                // Save previous solution.
                /*
                Graph::vertex_iterator vit, end;
                boost::tie(vit, end) = boost::vertices(*graph);
                for ( ; vit != end; ++vit) {
                  VertexData& vtx = (*graph)[*vit];
                  vtx.x_prev = vtx.x;
                  vtx.w1_prev = vtx.w1;
                  vtx.w2_prev = vtx.w2;
                }
                 */

                for (auto [id, vtx]: graph->get_vertices()) {
                    vtx.x_prev = vtx.x;
                    vtx.w1_prev = vtx.w1;
                    vtx.w2_prev = vtx.w2;
                }

                internal::dualStep(params, graph);
                internal::primalStep(params, graph);
                internal::extraGradientStep(params, graph);
            }

            float smoothnessCost(const Params &params, const Graph &graph) {
                float cost = 0.0f;

                auto fast_abs = [](float a) { return (a >= 0) ? a : -a; };

                for (const auto &[vertex_id_pair, edge]: graph.get_edges()) {
                    const VertexData &vtx_ii = graph.get_vertex(vertex_id_pair.first);
                    const VertexData &vtx_jj = graph.get_vertex(vertex_id_pair.second);

                    cv::Point2f xy_diff = vtx_ii.pos - vtx_jj.pos;
                    cost += edge.alpha * fast_abs(vtx_ii.x - vtx_jj.x -
                                                  vtx_ii.w1 * xy_diff.x - vtx_ii.w2 * xy_diff.y);
                    cost += edge.beta * fast_abs(vtx_ii.w1 - vtx_jj.w1) +
                            edge.beta * fast_abs(vtx_ii.w2 - vtx_jj.w2);
                }

                /*
                Graph::edge_iterator eit, end;
                boost::tie(eit, end) = boost::edges(graph);
                for (; eit != end; ++eit) {
                    const EdgeData &edge = graph[*eit];
                    const VertexData &vtx_ii = graph[boost::source(*eit, graph)];
                    const VertexData &vtx_jj = graph[boost::target(*eit, graph)];

                    cv::Point2f xy_diff = vtx_ii.pos - vtx_jj.pos;
                    cost += edge.alpha * fast_abs(vtx_ii.x - vtx_jj.x -
                                                  vtx_ii.w1 * xy_diff.x - vtx_ii.w2 * xy_diff.y);
                    cost += edge.beta * fast_abs(vtx_ii.w1 - vtx_jj.w1) +
                            edge.beta * fast_abs(vtx_ii.w2 - vtx_jj.w2);
                }
                 */

                return params.data_factor * cost;
            }

            float dataCost(const Params &params, const Graph &graph) {
                float cost = 0.0f;

                for (auto &[id, vtx]: graph.get_vertices()) {
                    float diff = (vtx.x - vtx.data_term) * vtx.data_weight;
                    diff = (diff > 0) ? diff : -diff;
                    cost += diff;
                }
                /*
                Graph::vertex_iterator vit, end;
                boost::tie(vit, end) = boost::vertices(graph);
                for (; vit != end; ++vit) {
                    const VertexData &vtx = graph[*vit];
                    float diff = (vtx.x - vtx.data_term) * vtx.data_weight;
                    diff = (diff > 0) ? diff : -diff;
                    cost += diff;
                }
                 */

                return cost;
            }

            namespace internal {

                void dualStep(const Params &params, Graph *graph) {
                    // Iterate over edges.
                    for (auto &[vertex_id_pair, cedge]: graph->get_edges()) {

                        VertexData &vtx_ii = graph->get_vertex(vertex_id_pair.first);
                        VertexData &vtx_jj = graph->get_vertex(vertex_id_pair.second);
                        EdgeData &edge = graph->get_edge(vertex_id_pair.first, vertex_id_pair.second);
                        // Update q1.
                        float K1x = edge.alpha * (vtx_ii.x_bar - vtx_jj.x_bar);
                        K1x -= edge.alpha * (vtx_ii.pos.x - vtx_jj.pos.x) * vtx_ii.w1_bar;
                        K1x -= edge.alpha * (vtx_ii.pos.y - vtx_jj.pos.y) * vtx_ii.w2_bar;
                        edge.q1 = proxNLTGV2Conj(params.step_q, edge.q1 + params.step_q * K1x);

                        // Update q2.
                        float K2x = edge.beta * (vtx_ii.w1_bar - vtx_jj.w1_bar);
                        edge.q2 = proxNLTGV2Conj(params.step_q, edge.q2 + params.step_q * K2x);

                        // Update q3.
                        float K3x = edge.beta * (vtx_ii.w2_bar - vtx_jj.w2_bar);
                        edge.q3 = proxNLTGV2Conj(params.step_q, edge.q3 + params.step_q * K3x);
                    }
                    /*
                    Graph::edge_iterator eit, end;
                    boost::tie(eit, end) = boost::edges(*graph);
                    for (; eit != end; ++eit) {
                        EdgeData &edge = (*graph)[*eit];
                        VertexData &vtx_ii = (*graph)[boost::source(*eit, *graph)];
                        VertexData &vtx_jj = (*graph)[boost::target(*eit, *graph)];

                        // Update q1.
                        float K1x = edge.alpha * (vtx_ii.x_bar - vtx_jj.x_bar);
                        K1x -= edge.alpha * (vtx_ii.pos.x - vtx_jj.pos.x) * vtx_ii.w1_bar;
                        K1x -= edge.alpha * (vtx_ii.pos.y - vtx_jj.pos.y) * vtx_ii.w2_bar;
                        edge.q1 = proxNLTGV2Conj(params.step_q, edge.q1 + params.step_q * K1x);

                        // Update q2.
                        float K2x = edge.beta * (vtx_ii.w1_bar - vtx_jj.w1_bar);
                        edge.q2 = proxNLTGV2Conj(params.step_q, edge.q2 + params.step_q * K2x);

                        // Update q3.
                        float K3x = edge.beta * (vtx_ii.w2_bar - vtx_jj.w2_bar);
                        edge.q3 = proxNLTGV2Conj(params.step_q, edge.q3 + params.step_q * K3x);
                    }
                     */
                }

                void primalStep(const Params &params, Graph *graph) {
                    // Iterate over edges.

                    for (const auto &[vertex_ids, edge]: graph->get_edges()) {
                        VertexData &vtx_ii = graph->get_vertex(vertex_ids.first);
                        VertexData &vtx_jj = graph->get_vertex(vertex_ids.second);

                        // Apply updates that require q1.
                        vtx_ii.x -= edge.q1 * params.step_x * edge.alpha;
                        vtx_jj.x += edge.q1 * params.step_x * edge.alpha;

                        vtx_ii.w1 += edge.q1 * params.step_x * edge.alpha *
                                     (vtx_ii.pos.x - vtx_jj.pos.x);

                        vtx_ii.w2 += edge.q1 * params.step_x * edge.alpha *
                                     (vtx_ii.pos.y - vtx_jj.pos.y);

                        // Apply updates that require q2.
                        vtx_ii.w1 -= edge.q2 * params.step_x * edge.beta;
                        vtx_jj.w1 += edge.q2 * params.step_x * edge.beta;

                        // Apply updates that require q3.
                        vtx_ii.w2 -= edge.q3 * params.step_x * edge.beta;
                        vtx_jj.w2 += edge.q3 * params.step_x * edge.beta;
                    }

                    /*
                    Graph::edge_iterator eit, end;
                    boost::tie(eit, end) = boost::edges(*graph);
                    for (; eit != end; ++eit) {
                        EdgeData &edge = (*graph)[*eit];
                        VertexData &vtx_ii = (*graph)[boost::source(*eit, *graph)];
                        VertexData &vtx_jj = (*graph)[boost::target(*eit, *graph)];

                        // Apply updates that require q1.
                        vtx_ii.x -= edge.q1 * params.step_x * edge.alpha;
                        vtx_jj.x += edge.q1 * params.step_x * edge.alpha;

                        vtx_ii.w1 += edge.q1 * params.step_x * edge.alpha *
                                     (vtx_ii.pos.x - vtx_jj.pos.x);

                        vtx_ii.w2 += edge.q1 * params.step_x * edge.alpha *
                                     (vtx_ii.pos.y - vtx_jj.pos.y);

                        // Apply updates that require q2.
                        vtx_ii.w1 -= edge.q2 * params.step_x * edge.beta;
                        vtx_jj.w1 += edge.q2 * params.step_x * edge.beta;

                        // Apply updates that require q3.
                        vtx_ii.w2 -= edge.q3 * params.step_x * edge.beta;
                        vtx_jj.w2 += edge.q3 * params.step_x * edge.beta;
                    }
                     */

                    for (auto &[id, cvtx]: graph->get_vertices()) {
                        VertexData &vtx = graph->get_vertex(id);
                        vtx.x = proxL1(params.x_min, params.x_max, params.step_x,
                                       params.data_factor * vtx.data_weight, vtx.x, vtx.data_term);
                    }

                    // Apply proximal operator to each vertex.
                    /*
                    Graph::vertex_iterator vit, vend;
                    boost::tie(vit, vend) = boost::vertices(*graph);
                    for (; vit != vend; ++vit) {
                        VertexData &vtx = (*graph)[*vit];
                        vtx.x = proxL1(params.x_min, params.x_max, params.step_x,
                                       params.data_factor * vtx.data_weight, vtx.x, vtx.data_term);
                    }
                    */

                }

                void extraGradientStep(const Params &params, Graph *graph) {
                    // Iteratate over vertices.

                    for (auto &[id, cvtx]: graph->get_vertices()) {
                        VertexData &vtx = graph->get_vertex(id);

                        float new_x_bar = vtx.x + params.theta * (vtx.x - vtx.x_prev);

                        // Project back onto the feasible set.
                        new_x_bar = (new_x_bar < params.x_min) ? params.x_min : new_x_bar;
                        new_x_bar = (new_x_bar > params.x_max) ? params.x_max : new_x_bar;
                        vtx.x_bar = new_x_bar;

                        vtx.w1_bar = vtx.w1 + params.theta * (vtx.w1 - vtx.w1_prev);
                        vtx.w2_bar = vtx.w2 + params.theta * (vtx.w2 - vtx.w2_prev);
                    }
                    /*
                    Graph::vertex_iterator vit, end;
                    boost::tie(vit, end) = boost::vertices(*graph);
                    for (; vit != end; ++vit) {
                        VertexData &vtx = (*graph)[*vit];
                        float new_x_bar = vtx.x + params.theta * (vtx.x - vtx.x_prev);

                        // Project back onto the feasible set.
                        new_x_bar = (new_x_bar < params.x_min) ? params.x_min : new_x_bar;
                        new_x_bar = (new_x_bar > params.x_max) ? params.x_max : new_x_bar;
                        vtx.x_bar = new_x_bar;

                        vtx.w1_bar = vtx.w1 + params.theta * (vtx.w1 - vtx.w1_prev);
                        vtx.w2_bar = vtx.w2 + params.theta * (vtx.w2 - vtx.w2_prev);
                    }
                */
                }

            }  // namespace internal

        }  // namespace nltgv2_l1_graph_regularizer

    }  // namespace optimizers

}  // namespace flame
