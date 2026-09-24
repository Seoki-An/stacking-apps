"""Offline dense RGB agreement filtering and provisional stone coverage accounting."""
from pathlib import Path
import inspect
import numpy as np
from scipy.spatial import cKDTree

from stacking_common.multiview.confidence import voxel_observations, filter_rgb_agreement, classify_reference_coverage
from stacking_common.multiview.dense import read_colmap_depth, backproject_colmap_depth
from stacking_common.rendering.pointcloud import render_clouds
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session, inside, read_json
from .hand_eye_rgb import statistics
from .storage import write_xyz_pcd


def validate_pair_sources(control, view_count):
    pairs=control.get('source_pairs',{})
    if len(pairs)!=2:
        raise ValueError('two disjoint RGB source pairs are required')
    indices=list(pairs.values())
    if any(len(p)!=2 or len(set(p))!=2 or any(type(v) is not int or not 0<=v<view_count for v in p) for p in indices):
        raise ValueError('invalid source pair view indices')
    if set(indices[0])&set(indices[1]):
        raise ValueError('source pairs must use disjoint RGB images')
    return pairs


def filter_dense_session(session, dense, output):
    import open3d as o3d
    session,dense,output=[Path(p).resolve() for p in (session,dense,output)]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (session,dense)):
        raise ValueError('confidence output must be separate from session and dense inputs')
    manifest=inspect_session(session);source=read_json(dense/'summary.json')
    pair_root=dense/'independent_pairs';control=read_json(pair_root/'summary.json')
    if (source.get('status')!='complete' or source.get('kind')!='dense_rgb_mvs_experiment'
            or source.get('session_sha256')!=digest(session/'session.json')
            or control.get('status')!='complete' or control.get('kind')!='disjoint_image_pair_control'
            or control.get('dense_summary_sha256')!=digest(dense/'summary.json')):
        raise ValueError('dense and disjoint-pair results must be complete and bound to this session')
    pairs=validate_pair_sources(control,len(manifest['views']))
    output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',kind='dense_rgb_confidence',schema_version=1,output_frame='base_link',
                session_sha256=digest(session/'session.json'),dense_sha256=digest(dense/'summary.json'),
                pair_control_sha256=digest(pair_root/'summary.json'),data_origin=manifest['data_origin'],
                source_sha256={'hand_eye_confidence.py':digest(__file__),'confidence.py':digest(inspect.getfile(filter_rgb_agreement))},
                accuracy_1mm_verified=False,production_fusion_accepted=False,input_hashes={})
    atomic_json(output/'summary.json',report)
    try:
        point_arrays=[];view_arrays=[];pixel_arrays=[];view_sources=[]
        low,high=np.array(source['region']['minimum']),np.array(source['region']['maximum'])
        for view in source['modes']['refined']['views']:
            index=view['view'];path=dense/'refined'/f'view_{index:03d}.npz'
            report['input_hashes'][str(path.relative_to(dense))]=digest(path)
            depth_path=dense/'refined/workspace/stereo/depth_maps'/(view['name']+'.geometric.bin')
            if digest(depth_path)!=view['depth_sha256']:
                raise ValueError('dense depth map hash mismatch')
            with np.load(path,allow_pickle=False) as a:
                points=a['points'];pixels=a['pixel_index'];matrix=a['matrix'];pose=a['base_link_T_camera'];shape=a['image_shape']
            depth=read_colmap_depth(depth_path)
            reconstructed,source_pixels=backproject_colmap_depth(depth,matrix,pose)
            selected=np.all((reconstructed>=low)&(reconstructed<=high),axis=1)
            if (tuple(shape)!=depth.shape or points.shape!=reconstructed[selected].shape
                    or not np.array_equal(pixels,source_pixels[selected])
                    or not np.allclose(points,reconstructed[selected],rtol=0,atol=1e-10)):
                raise ValueError('dense point/pixel provenance differs from source depth')
            point_arrays.append(points);pixel_arrays.append(pixels.astype(np.int32));view_arrays.append(np.full(len(points),index,np.uint16))
            view_sources.append(dict(view=index,file=str(path.relative_to(dense)),image_shape=shape.tolist(),depth_sha256=view['depth_sha256']))
        voxels=voxel_observations(np.concatenate(point_arrays),np.concatenate(view_arrays),np.concatenate(pixel_arrays))
        points=voxels['points']
        if len(points)!=source['modes']['refined']['voxel_roi_points']:
            raise ValueError('replayed voxel count differs from dense result')
        original_path=dense/'refined/rgb_dense_voxel_1mm.pcd'
        previous=np.asarray(o3d.io.read_point_cloud(str(original_path)).points)
        if previous.shape!=points.shape or not np.allclose(previous,points,atol=1e-7,rtol=0):
            raise ValueError('replayed voxel coordinates differ from dense result')
        report['input_hashes'][str(original_path.relative_to(dense))]=digest(original_path)
        pair_clouds=[]
        for name in pairs:
            path=inside(pair_root,name+'.pcd')
            report['input_hashes'][str(path.relative_to(dense))]=digest(path)
            pair_clouds.append(np.asarray(o3d.io.read_point_cloud(str(path)).points))
        print(f'Filtering {len(points):,} RGB voxel samples with disjoint-pair support',flush=True)
        quality=filter_rgb_agreement(points,pair_clouds)
        keep=quality['keep']
        root=session/'femto_before';reference_path=inside(root,read_json(root/'field_scan.json')['fused_file'])
        reference=np.asarray(o3d.io.read_point_cloud(str(reference_path)).points)
        reference=reference[np.all((reference>=low)&(reference<=high),axis=1)]
        region=source['region'];coeff=np.array(region['support_plane'])
        def mask(p,lo,hi):return (p[:,2]>p[:,:2]@coeff[:2]+coeff[2]+.008)&np.all((p[:,:2]>=lo)&(p[:,:2]<=hi),axis=1)
        stone_labels=np.full(len(points),-1,np.int16);coverage_labels=np.zeros(len(points),np.uint8);ref_distances=np.full(len(points),np.nan)
        groups=[];targets=[]
        for stone,(lo,hi) in enumerate(region['stone_boxes_xy']):
            region_mask=mask(points,lo,hi);ids=np.flatnonzero(region_mask&keep);target=reference[mask(reference,lo,hi)];targets.append(target)
            stone_labels[region_mask]=stone
            classified=classify_reference_coverage(points[ids],target)
            coverage_labels[ids]=classified['labels'];ref_distances[ids]=classified['reference_distance_m']
            before_dist=cKDTree(target).query(points[region_mask])[0]*1000
            reverse_before=cKDTree(points[region_mask]).query(target)[0] if region_mask.any() else np.full(len(target),np.inf)
            reverse_after=cKDTree(points[ids]).query(target)[0] if len(ids) else np.full(len(target),np.inf)
            groups.append(dict(stone=stone,input_points=int(region_mask.sum()),retained_points=len(ids),
                rejected_pair_disagreement=int(np.count_nonzero(region_mask&~quality['pair_agreement'])),
                rejected_local_support=int(np.count_nonzero(region_mask&quality['pair_agreement']&~quality['local_support'])),
                reference_overlap_points=int(np.count_nonzero(classified['labels']==1)),
                unresolved_surface_difference_points=int(np.count_nonzero(classified['labels']==2)),
                candidate_extension_points=int(np.count_nonzero(classified['labels']==3)),
                candidate_xy_cells=classified['candidate_xy_cells'],candidate_projected_area_mm2=classified['candidate_projected_area_mm2'],
                before_femto_mm=statistics(before_dist),after_femto_mm=statistics(classified['reference_distance_m']*1000),
                femto_fraction_within_3mm_before=float(np.mean(reverse_before<=.003)),femto_fraction_within_3mm_after=float(np.mean(reverse_after<=.003))))
        np.savez_compressed(output/'points_and_provenance.npz',**voxels,keep=keep,pair_agreement=quality['pair_agreement'],
                            local_support=quality['local_support'],pair_distances_m=quality['pair_distances_m'],
                            stone=stone_labels,coverage_class=coverage_labels,reference_distance_m=ref_distances)
        write_xyz_pcd(output/'retained_rgb_scene.pcd',points[keep])
        write_xyz_pcd(output/'rejected_rgb_scene.pcd',points[~keep])
        layers={'FEMTO reference':np.concatenate(targets)}
        for label,name in [(1,'stone_reference_overlap'),(2,'stone_unresolved_difference'),(3,'stone_candidate_extensions')]:
            selected=points[coverage_labels==label];write_xyz_pcd(output/(name+'.pcd'),selected);layers[name]=selected
        layers['rejected RGB stone-region samples']=points[(stone_labels>=0)&~keep]
        render_clouds(output/'comparison',layers,{},title='Filtered RGB: reference overlap / unresolved difference / candidate extensions')
        report.update(status='complete',point_count=int(keep.sum()),input_points=len(points),
            rejected_points=int((~keep).sum()),retained_stone_points=int(np.count_nonzero(keep&(stone_labels>=0))),
            groups=groups,gates=quality['gates'],voxel_size_m=.001,region=region,
            coverage_settings=dict(overlap_distance_m=.003,footprint_cell_m=.002,reference_footprint_padding_cells=1),
            view_sources=view_sources,pair_sources=pairs,
            provenance='CSR: offsets[i]:offsets[i+1] indexes observation_views/observation_pixels for point i; pixels index undistorted depth images in view_sources.',
            coverage_classes={'0':'background, unselected, or rejected (consult keep/stone)','1':'within 3mm of FEMTO','2':'farther than 3mm but inside padded reference XY footprint','3':'farther than 3mm and outside padded reference XY footprint; unverified extension'},
            limitations=['Gates are deterministic quality tests, not calibrated confidence probabilities or 1mm accuracy acceptance.',
            'Support from disjoint image pairs reduces some matching artifacts; all inputs still share intrinsics and refined pose assumptions.',
            'No FEMTO geometry is used for RGB filtering. Its data and existing ROI masks enter classification/evaluation only.',
            'A 2mm XY footprint is a top-view occupancy proxy, not camera visibility or 3D surface area.',
            'Candidate extensions may be lateral calibration error or shared artifacts; no extension is certified as new coverage.',
            'Conservative pair support can remove true surfaces visible to only one pair and thin details; rejected evidence is preserved.',
            'No ICP, coordinate adjustment, hole filling, smoothing, or reference/RGB blending is performed.'])
        atomic_json(output/'summary.json',report)
        return report
    except BaseException as error:
        report.update(status='failed',error=str(error));atomic_json(output/'summary.json',report);raise
