import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import random


def vis_gridpath_2d(n, paths, obs_nodes=None):
    """
    グリッドグラフ上に経路を描画する。

    Args:
        n (int):
        paths (List[List[int]]): 経路の集合。一つの経路はnodeのwaypointから構成される。
    Returns:
        None
    """
    node_colors = ['orange', 'cyan', 'greenyellow', 'violet']
    edge_colors = ['red', 'blue', 'green', 'blueviolet']
    G = nx.grid_2d_graph(n, n)
    mapping = {(i, j): i * n + j for i, j in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    pos = {i * n + j: (j, -i) for i in range(n) for j in range(n)}
    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos)

    if obs_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=obs_nodes, node_color='black', node_size=500)
        labels = {n: n for n in obs_nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, font_color='white')

    for (i, path) in enumerate(paths):
        edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange')
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=2)

        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color=node_colors[i % len(paths)])
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors[i % len(paths)], width=2)

    plt.axis('off')
    plt.show()



def vis_gridpath_3d(n, paths=None, obs_nodes=None, saveflg=False, savefilename: str =None):
    """
    3次元格子グラフ上に経路を描画する（ノード中央ラベル & 表示切替ボタン）
    - 直方体対応: n が int なら (n,n,n)、tuple/list なら (n_x,n_y,n_z)
    """
    # ====== 変更点(1): n を (n_x, n_y, n_z) に正規化 ======
    if isinstance(n, (tuple, list)):
        n_x, n_y, n_z = n
    else:
        n_x = n_y = n_z = n

    N = n_x * n_y * n_z

    # --- ノード定義（z層ごと連番） ---
    pos = {}
    counter = 0
    for z in range(n_z):
        for y in range(n_y):
            for x in range(n_x):
                pos[counter] = (x, y, z)
                counter += 1

    # --- エッジ定義（6近傍） ---
    G = nx.Graph()
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                # ====== 変更点(2): idx の計算を n_x,n_y 版に ======
                idx = k * n_x * n_y + j * n_x + i
                for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                    ni, nj, nk = i + dx, j + dy, k + dz
                    # ====== 変更点(3): 境界判定を n_x,n_y,n_z 版に ======
                    if ni < n_x and nj < n_y and nk < n_z:
                        nidx = nk * n_x * n_y + nj * n_x + ni
                        G.add_edge(idx, nidx)

    # --- 座標 ---
    node_x = [pos[i][0] for i in pos]
    node_y = [pos[i][1] for i in pos]
    node_z = [pos[i][2] for i in pos]

    # --- エッジ座標 ---
    edge_x, edge_y, edge_z = [], [], []
    for e in G.edges():
        x0, y0, z0 = pos[e[0]]
        x1, y1, z1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # --- エッジ描画 ---
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none',
        name='Edges'
    )

    # --- ノードカラー決定 ---
    # ====== 変更点(4): n**3 -> N ======
    node_color = ['lightgray'] * N
    if obs_nodes:
        for obs in obs_nodes:
            if 0 <= obs < len(node_color):
                node_color[obs] = 'black'

    path_colors = ['red', 'blue', 'green', 'orange', 'violet', 'cyan']
    if paths:
        for i, path in enumerate(paths):
            col = path_colors[i % len(path_colors)]
            for node in path:
                node_color[node] = col

    # --- 灰色ノード（半透明）とその他ノード（不透明）を分けて描画 ---
    gray_nodes  = [i for i,c in enumerate(node_color) if c == 'lightgray']
    color_nodes = [i for i,c in enumerate(node_color) if c != 'lightgray']

    traces = [edge_trace]

    if gray_nodes:
        traces.append(go.Scatter3d(
            x=[node_x[i] for i in gray_nodes],
            y=[node_y[i] for i in gray_nodes],
            z=[node_z[i] for i in gray_nodes],
            mode='markers',
            marker=dict(size=9, color='lightgray', opacity=0.6),
            name='Normal Nodes',
            hovertext=[f'Node {i}' for i in gray_nodes],
            hoverinfo='text'
        ))

    if color_nodes:
        traces.append(go.Scatter3d(
            x=[node_x[i] for i in color_nodes],
            y=[node_y[i] for i in color_nodes],
            z=[node_z[i] for i in color_nodes],
            mode='markers',
            marker=dict(size=9, color=[node_color[i] for i in color_nodes], opacity=0.9),
            name='Colored Nodes',
            hovertext=[f'Node {i}' for i in color_nodes],
            hoverinfo='text'
        ))

    # --- 経路 ---
    if paths:
        for i, path in enumerate(paths):
            px, py, pz = [], [], []
            for node in path:
                x, y, z = pos[node]
                px.append(x); py.append(y); pz.append(z)
            traces.append(go.Scatter3d(
                x=px, y=py, z=pz,
                mode='lines+markers',
                line=dict(color=path_colors[i % len(path_colors)], width=6),
                marker=dict(size=10, color=path_colors[i % len(path_colors)]),
                name=f'Path {i}'
            ))

    # --- ノード番号（scene.annotations で完全中央に重ねる） ---
    def make_label_annotations():
        anns = []
        for i in range(len(node_x)):
            c = node_color[i]
            text_color = 'white' if c in ['red','blue','green','orange','violet','cyan','black'] else 'darkblue'
            anns.append(dict(
                showarrow=False,
                x=node_x[i], y=node_y[i], z=node_z[i],
                text=str(i),
                xanchor='center', yanchor='middle',
                font=dict(color=text_color, size=11),
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(0,0,0,0)'
            ))
        return anns

    annotations_on  = make_label_annotations()
    annotations_off = []

    # --- スケーリング設定 ---
    margin = 0.5
    # ====== 変更点(5): 軸rangeを各軸で分ける ======
    axis_range_x = [-margin, n_x - 1 + margin]
    axis_range_y = [-margin, n_y - 1 + margin]
    axis_range_z = [-margin, n_z - 1 + margin]

    # ついでにサイズは最大辺でスケール（見た目が自然）
    n_max = max(n_x, n_y, n_z)
    fig_size = max(600, min(1400, 100 + n_max * 100))

    fig = go.Figure(data=traces)
    fig.update_layout(
        # ====== 変更点(6): タイトル表示 ======
        title=f"3D Grid Graph ({n_x}x{n_y}x{n_z})",
        showlegend=True,
        legend=dict(itemsizing='constant'),
        scene=dict(
            xaxis=dict(title='X', range=axis_range_x, showbackground=True),
            yaxis=dict(title='Y', range=axis_range_y, showbackground=True),
            zaxis=dict(title='Z', range=axis_range_z, showbackground=True),
            # 直方体でも歪ませない（軸の実長に合わせる）
            aspectmode='data',
            annotations=annotations_off
        ),
        width=fig_size,
        height=fig_size,
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, y=1.08, xanchor="center", yanchor="bottom",
                buttons=[
                    dict(label="Hide Labels", method="relayout",
                         args=[{"scene.annotations": annotations_off}]),
                    dict(label="Show Labels", method="relayout",
                         args=[{"scene.annotations": annotations_on}]),
                ]
            )
        ]
    )

    # html保存
    if saveflg:
        savepath = os.path.join("results", savefilename)
        fig.write_html(savepath)

    fig.show()


def vis_gridpath_3d_only_cube(n, paths=None, obs_nodes=None, saveflg=False, savefilename: str =None):
    """
    3次元格子グラフ上に経路を描画する（ノード中央ラベル & 表示切替ボタン）
    - ラベルは scene.annotations で完全中央に重ね表示
    - 濃色ノード上の文字は白、灰色ノード上は濃紺
    - n に応じて描画範囲と図サイズを自動スケーリング
    - 灰色ノードは半透明（0.6）
    - 立方体上の格子点描画のみ対応（旧 vis_gridpath_3d() ）# 将来的に削除予定
    """
    # --- ノード定義（z層ごと連番） ---
    pos = {}
    counter = 0
    for z in range(n):
        for y in range(n):
            for x in range(n):
                pos[counter] = (x, y, z)
                counter += 1

    # --- エッジ定義（6近傍） ---
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = k * n * n + j * n + i
                for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
                    ni, nj, nk = i + dx, j + dy, k + dz
                    if ni < n and nj < n and nk < n:
                        nidx = nk * n * n + nj * n + ni
                        G.add_edge(idx, nidx)

    # --- 座標 ---
    node_x = [pos[i][0] for i in pos]
    node_y = [pos[i][1] for i in pos]
    node_z = [pos[i][2] for i in pos]

    # --- エッジ座標 ---
    edge_x, edge_y, edge_z = [], [], []
    for e in G.edges():
        x0, y0, z0 = pos[e[0]]
        x1, y1, z1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    # --- エッジ描画 ---
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='gray', width=2),
        hoverinfo='none',
        name='Edges'
    )

    # --- ノードカラー決定 ---
    node_color = ['lightgray'] * (n**3)
    if obs_nodes:
        for obs in obs_nodes:
            if 0 <= obs < len(node_color):
                node_color[obs] = 'black'

    path_colors = ['red', 'blue', 'green', 'orange', 'violet', 'cyan']
    if paths:
        for i, path in enumerate(paths):
            col = path_colors[i % len(path_colors)]
            for node in path:
                node_color[node] = col

    # --- 灰色ノード（半透明）とその他ノード（不透明）を分けて描画 ---
    gray_nodes  = [i for i,c in enumerate(node_color) if c == 'lightgray']
    color_nodes = [i for i,c in enumerate(node_color) if c != 'lightgray']

    traces = [edge_trace]

    if gray_nodes:
        traces.append(go.Scatter3d(
            x=[node_x[i] for i in gray_nodes],
            y=[node_y[i] for i in gray_nodes],
            z=[node_z[i] for i in gray_nodes],
            mode='markers',
            marker=dict(size=9, color='lightgray', opacity=0.6),
            # marker=dict(size=9, sizemode='diameter', color='lightgray', opacity=0.6),
            name='Normal Nodes',
            hovertext=[f'Node {i}' for i in gray_nodes],
            hoverinfo='text'
        ))

    if color_nodes:
        traces.append(go.Scatter3d(
            x=[node_x[i] for i in color_nodes],
            y=[node_y[i] for i in color_nodes],
            z=[node_z[i] for i in color_nodes],
            mode='markers',
            marker=dict(size=9, color=[node_color[i] for i in color_nodes], opacity=0.9),
            # marker=dict(size=9, sizemode='diameter', color=[node_color[i] for i in color_nodes], opacity=0.9),
            name='Colored Nodes',
            hovertext=[f'Node {i}' for i in color_nodes],
            hoverinfo='text'
        ))

    # --- 経路 ---
    if paths:
        for i, path in enumerate(paths):
            px, py, pz = [], [], []
            for node in path:
                x, y, z = pos[node]
                px.append(x); py.append(y); pz.append(z)
            traces.append(go.Scatter3d(
                x=px, y=py, z=pz,
                mode='lines+markers',
                line=dict(color=path_colors[i % len(path_colors)], width=6),
                marker=dict(size=10, color=path_colors[i % len(path_colors)]),
                # marker=dict(size=10, sizemode='diameter', color=path_colors[i % len(path_colors)]),
                name=f'Path {i}'
            ))

    # --- ノード番号（scene.annotations で完全中央に重ねる） ---
    def make_label_annotations():
        anns = []
        for i in range(len(node_x)):
            c = node_color[i]
            text_color = 'white' if c in ['red','blue','green','orange','violet','cyan','black'] else 'darkblue'
            anns.append(dict(
                showarrow=False,
                x=node_x[i], y=node_y[i], z=node_z[i],
                text=str(i),
                xanchor='center', yanchor='middle',
                font=dict(color=text_color, size=11),
                bgcolor='rgba(0,0,0,0)',  # 背景なし（必要なら半透明背景も可）
                bordercolor='rgba(0,0,0,0)'
            ))
        return anns

    annotations_on  = make_label_annotations()
    annotations_off = []  # 非表示時

    # --- スケーリング設定 ---
    margin = 0.5
    axis_range = [-margin, n - 1 + margin]
    fig_size = max(600, min(1400, 100 + n * 100))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"3D Grid Graph (n={n})",
        showlegend=True,
        legend=dict(itemsizing='constant'),
        scene=dict(
            xaxis=dict(title='X', range=axis_range, showbackground=True),
            yaxis=dict(title='Y', range=axis_range, showbackground=True),
            zaxis=dict(title='Z', range=axis_range, showbackground=True),
            aspectmode='cube',
            annotations=annotations_off  # 初期状態：表示
        ),
        width=fig_size,
        height=fig_size,
        margin=dict(l=0, r=0, b=0, t=30),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.5, y=1.08, xanchor="center", yanchor="bottom",
                buttons=[
                    dict(
                        label="Hide Labels",
                        method="relayout",
                        args=[{"scene.annotations": annotations_off}]
                    ),
                    dict(
                        label="Show Labels",
                        method="relayout",
                        args=[{"scene.annotations": annotations_on}]
                    ),
                ]
            )
        ]
    )

    # html保存
    if saveflg:
        savepath = os.path.join("results", savefilename)
        fig.write_html(savepath)

    fig.show()
