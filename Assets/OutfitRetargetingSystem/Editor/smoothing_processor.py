"""
頂点グループスムージング処理を並列実行するスクリプト
Blenderのbpy依存を排除し、サブプロセスとして実行可能

このスクリプトは以下の2つのモードで動作する
1. 単一頂点グループモード: 1つの頂点グループのスムージング処理
2. マルチグループモード: 複数の頂点グループを並列処理

各プロセスでcKDTreeを構築して近傍検索を行うため、
neighbors_cacheを共有せずメモリ消費を抑制する。

使用方法:
# 単一グループモード
python smoothing_processor.py input_data.npz --max-workers 4

# マルチグループモード（複数の頂点グループを並列処理）
python smoothing_processor.py input_data.npz --multi-group --max-workers 4
"""

import numpy as np
import argparse
import sys
import os
import time
from pathlib import Path
import multiprocessing as mp
from multiprocessing import freeze_support, Pool
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional, Any

# Windows環境でのマルチプロセス問題を防ぐためのタイムアウト設定（秒）
DEFAULT_WORKER_TIMEOUT = 600  # 10分


def gaussian_weights(distances, sigma):
    """ガウシアン減衰による重み計算"""
    return np.exp(-(distances ** 2) / (2 * sigma ** 2))


def linear_weights(distances, radius):
    """線形減衰による重み計算"""
    return np.maximum(0, 1.0 - distances / radius)


def process_vertex_batch(args):
    """
    頂点のバッチに対してスムージング処理を適用
    
    Parameters:
        args: tuple of (start_idx, end_idx, vertex_coords, current_weights, 
                       use_distance_weighting, gaussian_falloff, 
                       smoothing_radius, sigma)
    
    Returns:
        tuple: (start_idx, end_idx, smoothed_weights)
    """
    (start_idx, end_idx, vertex_coords, current_weights, 
     use_distance_weighting, gaussian_falloff, 
     smoothing_radius, sigma) = args
    
    num_vertices = end_idx - start_idx
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)
    
    # プロセス内でcKDTreeを構築
    kdtree = cKDTree(vertex_coords)
    
    for local_i in range(num_vertices):
        i = start_idx + local_i
        # その場で近傍を検索
        neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)
        
        if len(neighbor_indices) > 1:  # 自分自身以外の近傍が存在する場合
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
            # 近傍頂点への距離を計算
            neighbor_coords = vertex_coords[neighbor_indices]
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
            
            # 近傍頂点のウェイト値を取得
            neighbor_weights = current_weights[neighbor_indices]
            
            if use_distance_weighting:
                if gaussian_falloff:
                    # ガウシアン減衰による重み計算
                    weights = gaussian_weights(distances, sigma)
                else:
                    # 線形減衰による重み計算
                    weights = linear_weights(distances, smoothing_radius)
                
                # 重み付き平均を計算
                weights_sum = np.sum(weights)
                if weights_sum > 0.001:
                    smoothed_weights[local_i] = neighbor_weights @ weights / weights_sum
                else:
                    smoothed_weights[local_i] = current_weights[i]
            else:
                # 従来の単純平均
                smoothed_weights[local_i] = np.mean(neighbor_weights)
        else:
            # 近傍頂点が自分だけの場合は元の値を保持
            smoothed_weights[local_i] = current_weights[i]
    
    return (start_idx, end_idx, smoothed_weights)


def apply_smoothing_sequential(vertex_coords, current_weights, kdtree,
                               smoothing_radius, use_distance_weighting, 
                               gaussian_falloff):
    """
    シングルスレッドでスムージング処理を実行（マルチグループモード用）
    
    Parameters:
        vertex_coords: 頂点座標のnumpy配列 (N, 3)
        current_weights: 現在のウェイト値のnumpy配列 (N,)
        kdtree: cKDTreeオブジェクト（近傍検索用）
        smoothing_radius: スムージング半径
        use_distance_weighting: 距離重み付けを使用するか
        gaussian_falloff: ガウシアン減衰を使用するか
    
    Returns:
        numpy.ndarray: スムージング後のウェイト配列
    """
    num_vertices = len(vertex_coords)
    sigma = smoothing_radius / 3.0
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)
    
    for i in range(num_vertices):
        # その場で近傍を検索
        neighbor_indices = kdtree.query_ball_point(vertex_coords[i], smoothing_radius)
        
        if len(neighbor_indices) > 1:
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int32)
            neighbor_coords = vertex_coords[neighbor_indices]
            distances = np.linalg.norm(neighbor_coords - vertex_coords[i], axis=1)
            neighbor_weights = current_weights[neighbor_indices]
            
            if use_distance_weighting:
                if gaussian_falloff:
                    weights = gaussian_weights(distances, sigma)
                else:
                    weights = linear_weights(distances, smoothing_radius)
                
                weights_sum = np.sum(weights)
                if weights_sum > 0.001:
                    smoothed_weights[i] = neighbor_weights @ weights / weights_sum
                else:
                    smoothed_weights[i] = current_weights[i]
            else:
                smoothed_weights[i] = np.mean(neighbor_weights)
        else:
            smoothed_weights[i] = current_weights[i]
    
    return smoothed_weights


def apply_smoothing_multithread(vertex_coords, current_weights,
                                smoothing_radius, use_distance_weighting, 
                                gaussian_falloff, max_workers=None):
    """
    マルチプロセスでスムージング処理を実行
    
    Parameters:
        vertex_coords: 頂点座標のnumpy配列 (N, 3)
        current_weights: 現在のウェイト値のnumpy配列 (N,)
        smoothing_radius: スムージング半径
        use_distance_weighting: 距離重み付けを使用するか
        gaussian_falloff: ガウシアン減衰を使用するか
        max_workers: 最大ワーカー数
    
    Returns:
        numpy.ndarray: スムージング後のウェイト配列
    """
    num_vertices = len(vertex_coords)
    sigma = smoothing_radius / 3.0
    
    if max_workers is None:
        max_workers = os.cpu_count()
    
    # バッチサイズを計算（最低でも1000頂点/バッチ）
    min_batch_size = 1000
    batch_size = max(min_batch_size, num_vertices // (max_workers * 4))
    
    # バッチに分割（neighbors_cacheなし、各プロセスでcKDTreeを構築）
    batches = []
    for start_idx in range(0, num_vertices, batch_size):
        end_idx = min(start_idx + batch_size, num_vertices)
        batches.append((
            start_idx, end_idx, vertex_coords, current_weights,
            use_distance_weighting, gaussian_falloff,
            smoothing_radius, sigma
        ))
    
    print(f"  頂点を{len(batches)}個のバッチに分割 (バッチサイズ: {batch_size})")
    print(f"  {max_workers}個のプロセスで並列処理を開始...")
    print(f"  （各プロセスでcKDTreeを構築）")
    
    # 並列処理を実行
    smoothed_weights = np.zeros(num_vertices, dtype=np.float32)
    
    with Pool(processes=max_workers) as pool:
        # imap_unorderedで非同期的に結果を取得
        completed = 0
        for result in pool.imap_unordered(process_vertex_batch, batches):
            start_idx, end_idx, batch_smoothed_weights = result
            smoothed_weights[start_idx:end_idx] = batch_smoothed_weights
            completed += 1
            
            if completed % max(1, len(batches) // 10) == 0:
                progress = (completed / len(batches)) * 100
                print(f"  進捗: {completed}/{len(batches)} バッチ完了 ({progress:.1f}%)")
    
    return smoothed_weights


def process_single_vertex_group(group_data: Dict[str, Any], vertex_coords: np.ndarray,
                                 smoothing_radius: float, iteration: int,
                                 use_distance_weighting: bool, gaussian_falloff: bool) -> Dict[str, Any]:
    """
    単一の頂点グループのスムージング処理を実行
    
    Parameters:
        group_data: 頂点グループのデータ
        vertex_coords: 頂点座標
        smoothing_radius: スムージング半径
        iteration: イテレーション回数
        use_distance_weighting: 距離重み付けを使用するか
        gaussian_falloff: ガウシアン減衰を使用するか
    
    Returns:
        Dict: 処理結果
    """
    group_name = group_data['group_name']
    original_weights = group_data['original_weights']
    
    # すべて0の場合はスキップ
    if not np.any(original_weights > 0):
        return {
            'group_name': group_name,
            'smoothed_weights': original_weights,
            'skipped': True
        }
    
    # プロセス内でcKDTreeを構築
    kdtree = cKDTree(vertex_coords)
    
    # スムージング処理を実行
    smoothed_weights = np.copy(original_weights)
    
    for iter_idx in range(iteration):
        smoothed_weights = apply_smoothing_sequential(
            vertex_coords=vertex_coords,
            current_weights=smoothed_weights,
            kdtree=kdtree,
            smoothing_radius=smoothing_radius,
            use_distance_weighting=use_distance_weighting,
            gaussian_falloff=gaussian_falloff
        )
    
    return {
        'group_name': group_name,
        'smoothed_weights': smoothed_weights,
        'skipped': False
    }


def process_vertex_group_worker(args: Tuple) -> Dict[str, Any]:
    """
    マルチプロセスワーカー関数: 1つの頂点グループを処理
    
    Parameters:
        args: (group_data, vertex_coords, smoothing_radius, 
               iteration, use_distance_weighting, gaussian_falloff)
    
    Returns:
        Dict: 処理結果
    """
    (group_data, vertex_coords, smoothing_radius,
     iteration, use_distance_weighting, gaussian_falloff) = args
    
    return process_single_vertex_group(
        group_data, vertex_coords,
        smoothing_radius, iteration, use_distance_weighting, gaussian_falloff
    )


def process_multiple_vertex_groups(vertex_coords: np.ndarray,
                                    groups_data: List[Dict[str, Any]],
                                    mask_weights: np.ndarray,
                                    smoothing_radius: float,
                                    iteration: int,
                                    use_distance_weighting: bool,
                                    gaussian_falloff: bool,
                                    max_workers: int = None) -> Dict[str, Dict[str, Any]]:
    """
    複数の頂点グループを並列処理
    
    Parameters:
        vertex_coords: 頂点座標のnumpy配列 (N, 3)
        groups_data: 頂点グループデータのリスト
        mask_weights: マスクウェイト（合成に使用、Noneの場合は合成しない）
        smoothing_radius: スムージング半径
        iteration: イテレーション回数
        use_distance_weighting: 距離重み付けを使用するか
        gaussian_falloff: ガウシアン減衰を使用するか
        max_workers: 最大ワーカー数
    
    Returns:
        Dict: {グループ名: {final_weights, skipped, ...}}
    """
    if max_workers is None:
        max_workers = os.cpu_count() - 1
    if max_workers < 1:
        max_workers = 1
    
    print(f"\n複数頂点グループの並列処理を開始")
    print(f"  頂点グループ数: {len(groups_data)}")
    print(f"  最大ワーカー数: {max_workers}")
    print(f"  頂点数: {len(vertex_coords)}")
    print(f"  スムージング半径: {smoothing_radius}")
    print(f"  イテレーション: {iteration}")
    print(f"  （各プロセスでcKDTreeを構築）")
    
    # 処理対象のグループを準備（ウェイトが0でないもの）
    valid_groups = []
    skipped_groups = []
    
    for group_data in groups_data:
        if np.any(group_data['original_weights'] > 0):
            valid_groups.append(group_data)
        else:
            skipped_groups.append(group_data['group_name'])
    
    if skipped_groups:
        print(f"  スキップされたグループ（全ウェイト0）: {len(skipped_groups)}")
    
    print(f"  処理対象グループ数: {len(valid_groups)}")
    
    results = {}
    
    # スキップされたグループの結果を追加
    for group_name in skipped_groups:
        # オリジナルのウェイトを見つける
        for gd in groups_data:
            if gd['group_name'] == group_name:
                results[group_name] = {
                    'final_weights': gd['original_weights'].copy(),
                    'smoothed_weights': gd['original_weights'].copy(),
                    'skipped': True
                }
                break
    
    if not valid_groups:
        print("  処理対象のグループがありません")
        return results
    
    # ワーカー用の引数を準備（neighbors_cacheなし）
    worker_args = [
        (group_data, vertex_coords, smoothing_radius,
         iteration, use_distance_weighting, gaussian_falloff)
        for group_data in valid_groups
    ]
    
    # 並列処理を実行
    process_start = time.time()
    
    # フォールバック用にworker_argsを保持
    failed_groups = []
    
    # グループ名から引数へのマッピングを作成
    group_name_to_args = {args[0]['group_name']: args for args in worker_args}

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    
    try:
        with Pool(processes=min(max_workers, len(valid_groups))) as pool:
            # async_resultを使用してタイムアウト付きで結果を取得
            async_result = pool.map_async(process_vertex_group_worker, worker_args)
            
            try:
                # タイムアウト付きで結果を取得
                all_results = async_result.get(timeout=DEFAULT_WORKER_TIMEOUT)
                
                completed = 0
                for result in all_results:
                    group_name = result['group_name']
                    
                    # 対応するoriginal_weightsを取得
                    original_weights = None
                    for gd in groups_data:
                        if gd['group_name'] == group_name:
                            original_weights = gd['original_weights']
                            break
                    
                    if not result['skipped'] and original_weights is not None:
                        smoothed_weights = result['smoothed_weights']
                        
                        # mask_weightsがNoneでない場合のみ合成処理を行う
                        if mask_weights is not None:
                            final_weights = np.zeros(len(vertex_coords), dtype=np.float32)
                            
                            for i in range(len(vertex_coords)):
                                blend_factor = mask_weights[i]
                                final_weights[i] = original_weights[i] * (1.0 - blend_factor) + smoothed_weights[i] * blend_factor
                            
                            results[group_name] = {
                                'final_weights': final_weights,
                                'smoothed_weights': smoothed_weights,
                                'skipped': False
                            }
                        else:
                            # マスクなしの場合はスムージング結果をそのまま使用
                            results[group_name] = {
                                'final_weights': smoothed_weights,
                                'smoothed_weights': smoothed_weights,
                                'skipped': False
                            }
                    else:
                        results[group_name] = {
                            'final_weights': original_weights if original_weights is not None else result['smoothed_weights'],
                            'smoothed_weights': result['smoothed_weights'],
                            'skipped': result['skipped']
                        }
                    
                    completed += 1
                    progress = (completed / len(valid_groups)) * 100
                    print(f"  グループ処理完了: {group_name} ({completed}/{len(valid_groups)}, {progress:.1f}%)")
                    
            except mp.TimeoutError:
                print(f"  警告: 並列処理全体がタイムアウトしました。残りのグループをシングルスレッドで処理します。")
                # 処理されていないグループを特定
                processed_groups = set(results.keys())
                for args in worker_args:
                    group_name = args[0]['group_name']
                    if group_name not in processed_groups:
                        failed_groups.append(args)
                        
    except Exception as e:
        print(f"  警告: multiprocessing.Poolでエラーが発生: {e}")
        import traceback
        traceback.print_exc()
        # 処理されていないグループを特定
        processed_groups = set(results.keys())
        for args in worker_args:
            group_name = args[0]['group_name']
            if group_name not in processed_groups:
                failed_groups.append(args)
    
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
    os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count())
    os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())
    
    # フォールバック処理: 失敗したグループをシングルスレッドで処理
    if failed_groups:
        print(f"\n  フォールバック処理を開始: {len(failed_groups)}グループ")
        for args in failed_groups:
            group_data = args[0]
            group_name = group_data['group_name']
            try:
                print(f"    処理中: {group_name}")
                result = process_vertex_group_worker(args)
                
                original_weights = None
                for gd in groups_data:
                    if gd['group_name'] == group_name:
                        original_weights = gd['original_weights']
                        break
                
                if not result['skipped'] and original_weights is not None:
                    smoothed_weights = result['smoothed_weights']
                    
                    if mask_weights is not None:
                        final_weights = np.zeros(len(vertex_coords), dtype=np.float32)
                        for i in range(len(vertex_coords)):
                            blend_factor = mask_weights[i]
                            final_weights[i] = original_weights[i] * (1.0 - blend_factor) + smoothed_weights[i] * blend_factor
                        results[group_name] = {
                            'final_weights': final_weights,
                            'smoothed_weights': smoothed_weights,
                            'skipped': False
                        }
                    else:
                        results[group_name] = {
                            'final_weights': smoothed_weights,
                            'smoothed_weights': smoothed_weights,
                            'skipped': False
                        }
                else:
                    results[group_name] = {
                        'final_weights': original_weights if original_weights is not None else result['smoothed_weights'],
                        'smoothed_weights': result['smoothed_weights'],
                        'skipped': result['skipped']
                    }
                print(f"    完了: {group_name}")
            except Exception as e:
                print(f"    エラー: グループ '{group_name}' のフォールバック処理も失敗: {e}")
                # 最終手段: オリジナルのウェイトを使用
                for gd in groups_data:
                    if gd['group_name'] == group_name:
                        results[group_name] = {
                            'final_weights': gd['original_weights'].copy(),
                            'smoothed_weights': gd['original_weights'].copy(),
                            'skipped': True
                        }
                        break
    
    process_time = time.time() - process_start
    print(f"\n  並列処理完了: {process_time:.2f}秒")
    
    return results


def main_single_group():
    """単一頂点グループモードのメイン処理"""
    parser = argparse.ArgumentParser(description='頂点グループスムージング処理（並列実行）')
    parser.add_argument('input_file', type=str, help='入力データファイル (.npz)')
    parser.add_argument('--max-workers', type=int, default=None, 
                       help='最大ワーカー数（デフォルト: CPU数）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("頂点グループスムージング処理を開始（単一グループモード）")
    print("="*60)
    
    start_time = time.time()
    
    # データを読み込み
    print(f"\n入力ファイル: {args.input_file}")
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        sys.exit(1)
    
    print("データを読み込み中...")
    load_start = time.time()
    data = np.load(args.input_file, allow_pickle=True)
    
    vertex_coords = data['vertex_coords']
    current_weights = data['current_weights']
    smoothing_radius = float(data['smoothing_radius'])
    iteration = int(data['iteration'])
    use_distance_weighting = bool(data['use_distance_weighting'])
    gaussian_falloff = bool(data['gaussian_falloff'])
    
    print(f"  頂点数: {len(vertex_coords)}")
    print(f"  スムージング半径: {smoothing_radius}")
    print(f"  イテレーション: {iteration}")
    print(f"  距離重み付け: {use_distance_weighting}")
    print(f"  ガウシアン減衰: {gaussian_falloff}")
    print(f"読み込み完了: {time.time() - load_start:.2f}秒")
    
    # スムージング処理を実行（イテレーション）
    # 注: 各プロセスでcKDTreeを構築するため、neighbors_cacheは使用しない
    smoothed_weights = np.copy(current_weights)
    
    for iter_idx in range(iteration):
        print(f"\nイテレーション {iter_idx + 1}/{iteration}")
        iter_start = time.time()
        
        smoothed_weights = apply_smoothing_multithread(
            vertex_coords=vertex_coords,
            current_weights=smoothed_weights,
            smoothing_radius=smoothing_radius,
            use_distance_weighting=use_distance_weighting,
            gaussian_falloff=gaussian_falloff,
            max_workers=args.max_workers
        )
        
        print(f"  イテレーション完了: {time.time() - iter_start:.2f}秒")
    
    # 結果を保存
    print("\n結果を保存中...")
    save_start = time.time()
    output_path = input_path.parent / f"{input_path.stem}_result.npz"
    
    np.savez_compressed(
        output_path,
        smoothed_weights=smoothed_weights
    )
    
    print(f"  出力ファイル: {output_path}")
    print(f"保存完了: {time.time() - save_start:.2f}秒")
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"処理が正常に完了しました (合計時間: {total_time:.2f}秒)")
    print("="*60)
    
    return 0


def main_multi_group():
    """複数頂点グループモードのメイン処理"""
    parser = argparse.ArgumentParser(description='複数頂点グループスムージング処理（並列実行）')
    parser.add_argument('input_file', type=str, help='入力データファイル (.npz)')
    parser.add_argument('--multi-group', action='store_true', help='マルチグループモード')
    parser.add_argument('--max-workers', type=int, default=None, 
                       help='最大ワーカー数（デフォルト: CPU数）')
    
    args = parser.parse_args()
    
    print("="*60)
    print("頂点グループスムージング処理を開始（マルチグループモード）")
    print("="*60)
    
    start_time = time.time()
    
    # データを読み込み
    print(f"\n入力ファイル: {args.input_file}")
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input_file}")
        sys.exit(1)
    
    print("データを読み込み中...")
    load_start = time.time()
    data = np.load(args.input_file, allow_pickle=True)
    
    vertex_coords = data['vertex_coords']
    smoothing_radius = float(data['smoothing_radius'])
    use_distance_weighting = bool(data['use_distance_weighting'])
    gaussian_falloff = bool(data['gaussian_falloff'])
    
    # first_groupの処理（マスクなしスムージング）
    first_group_name = str(data['first_group_name']) if 'first_group_name' in data else None
    first_group_weights = data['first_group_weights'] if 'first_group_weights' in data else None
    first_group_iteration = int(data['first_group_iteration']) if 'first_group_iteration' in data else 1
    
    # target_groupsの処理（マスク付きスムージング）
    target_group_iteration = int(data['target_group_iteration']) if 'target_group_iteration' in data else 3
    mask_weights = data['mask_weights'] if 'mask_weights' in data else None
    
    # 頂点グループデータを読み込み
    group_names = list(data['group_names']) if 'group_names' in data else []
    groups_data = []
    
    for group_name in group_names:
        # キー名からグループデータを取得
        key = f"group_{group_name}"
        if key in data:
            groups_data.append({
                'group_name': group_name,
                'original_weights': data[key]
            })
        else:
            print(f"  警告: グループ '{group_name}' のデータが見つかりません")
    
    print(f"  頂点数: {len(vertex_coords)}")
    print(f"  スムージング半径: {smoothing_radius}")
    print(f"  距離重み付け: {use_distance_weighting}")
    print(f"  ガウシアン減衰: {gaussian_falloff}")
    if first_group_name:
        print(f"  最初のグループ: {first_group_name} (iteration={first_group_iteration})")
    print(f"  追加グループ数: {len(groups_data)} (iteration={target_group_iteration})")
    print(f"読み込み完了: {time.time() - load_start:.2f}秒")
    
    results = {}
    
    # === first_groupの処理（マスクなしスムージング） ===
    if first_group_name is not None and first_group_weights is not None:
        print(f"\n=== 最初のグループ '{first_group_name}' を処理中 ===")
        
        if np.any(first_group_weights > 0):
            # first_group用にcKDTreeを構築
            print(f"  cKDTreeを構築中...")
            kdtree = cKDTree(vertex_coords)
            
            smoothed_weights = np.copy(first_group_weights)
            
            for iter_idx in range(first_group_iteration):
                print(f"  イテレーション {iter_idx + 1}/{first_group_iteration}")
                smoothed_weights = apply_smoothing_sequential(
                    vertex_coords=vertex_coords,
                    current_weights=smoothed_weights,
                    kdtree=kdtree,
                    smoothing_radius=smoothing_radius,
                    use_distance_weighting=use_distance_weighting,
                    gaussian_falloff=gaussian_falloff
                )
            
            results[first_group_name] = {
                'final_weights': smoothed_weights,
                'smoothed_weights': smoothed_weights,
                'skipped': False
            }
            print(f"  最初のグループ処理完了")
        else:
            results[first_group_name] = {
                'final_weights': first_group_weights.copy(),
                'smoothed_weights': first_group_weights.copy(),
                'skipped': True
            }
            print(f"  最初のグループはスキップ（全ウェイト0）")
    
    # === target_groupsの並列処理（マスク付きスムージング） ===
    if groups_data:
        print(f"\n=== 追加グループを並列処理中 ({len(groups_data)}グループ) ===")
        
        # 追加グループ用のマスクウェイトを計算
        final_mask_weights = None
        
        if first_group_name is not None and first_group_name in results:
            # first_groupのスムージング結果とオリジナル値を取得
            first_group_smoothed = results[first_group_name]['smoothed_weights']
            
            if mask_weights is not None:
                # mask_weightsが指定されている場合、first_groupのスムージング結果をマスクウェイトでマスク
                final_mask_weights = np.zeros(len(vertex_coords), dtype=np.float32)
                for i in range(len(vertex_coords)):
                    final_mask_weights[i] = first_group_smoothed[i] * mask_weights[i]
                print(f"  マスクウェイト: first_group '{first_group_name}' のスムージング結果をマスクウェイトでマスク")
            else:
                # mask_weightsがNoneの場合、first_groupのスムージング結果をそのまま使用
                final_mask_weights = first_group_smoothed
                print(f"  マスクウェイト: first_group '{first_group_name}' のスムージング結果を使用")
        elif mask_weights is not None:
            # first_groupがない場合はmask_weightsをそのまま使用
            final_mask_weights = mask_weights
            print(f"  マスクウェイト: 入力されたmask_weightsを使用")
        
        target_results = process_multiple_vertex_groups(
            vertex_coords=vertex_coords,
            groups_data=groups_data,
            mask_weights=final_mask_weights,
            smoothing_radius=smoothing_radius,
            iteration=target_group_iteration,
            use_distance_weighting=use_distance_weighting,
            gaussian_falloff=gaussian_falloff,
            max_workers=args.max_workers
        )
        
        # 結果をマージ
        results.update(target_results)
    
    # 結果を保存
    print("\n結果を保存中...")
    save_start = time.time()
    output_path = input_path.parent / f"{input_path.stem}_result.npz"
    
    # 保存用のデータを準備
    save_data = {
        'group_names': np.array(list(results.keys()))
    }
    
    for group_name, result in results.items():
        save_data[f"final_{group_name}"] = result['final_weights']
        save_data[f"smoothed_{group_name}"] = result['smoothed_weights']
        save_data[f"skipped_{group_name}"] = result['skipped']
    
    np.savez_compressed(output_path, **save_data)
    
    print(f"  出力ファイル: {output_path}")
    print(f"保存完了: {time.time() - save_start:.2f}秒")
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"処理が正常に完了しました (合計時間: {total_time:.2f}秒)")
    print("="*60)
    
    return 0


def main():
    """メインエントリーポイント"""
    # 引数をチェックしてモードを決定
    if '--multi-group' in sys.argv:
        return main_multi_group()
    else:
        return main_single_group()


if __name__ == '__main__':
    # Windows環境でのマルチプロセス問題を防ぐ
    freeze_support()
    
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\nエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
