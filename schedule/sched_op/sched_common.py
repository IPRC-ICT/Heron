import tvm
import tvm.te as te
from Heron.utils import *
from Heron.schedule.primitives import *
from .ana_common import *

class schedOp:
    def __init__(self, name):
        self.name = name

    def tileAxes(self, s, stage_name, ctx, axes):
        stage = mapNametoStage(s, stage_name)
        keys = []; outer = []; inner = [] 
        for ax in axes:
            key = genKey("P", stage_name, str(ax.var.name), param_name = self.name)
            up = ctx.knob_manager.get_axis_extent(s, stage_name, ax.var.name)
            ctx.knob_manager.define_value(key, 1, up, 1, True)
            ctx.knob_manager.addRawCandidates(key, 
                    getDefSplitCandidates(s, stage_name, ax.var.name, ctx.knob_manager))
            tile_size = ctx.knob_manager.get_val(key)
            xo, xi = split(ctx, stage, ax, key, nparts = tile_size)
            keys.append(key)
            outer.append(xo); inner.append(xi)
        return outer, inner, keys

    def tileUnderKeys(self, stage, axes, keys, is_factor, ctx):
        outer = []; inner = []
        for idx, ax in enumerate(axes):
            val = ctx.knob_manager.get_val(keys[idx])
            if is_factor:
                axo, axi = split(ctx, stage, ax, keys[idx], factor = val)
            else:
                axo, axi = split(ctx, stage, ax, keys[idx], nparts = val)
            outer.append(axo); inner.append(axi)
        return outer, inner

    def findUnscheduledAxes(self, ctx, stage_name, axes):
        spatial_ = []; reduce_ = []
        for ax in axes:
            key = genKey("L", stage_name, str(ax.var.name))
            if key in ctx.scheduled_axes:
                continue
            if ax.iter_type == ax.DataPar:
                spatial_.append(ax)
            else:
                reduce_.append(ax)
        return spatial_, reduce_

    def findUnscheduledSpatialAxes(self, ctx, stage_name, axes):
        res = []
        for ax in axes:
            key = genKey("L", stage_name, str(ax.var.name))
            if key in ctx.scheduled_axes or ax.iter_type != ax.DataPar:
                continue
            res.append(ax)
        return res

    def formTobindAxes(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        axes = stage.leaf_iter_vars
        res = []
        for ax in axes:
            key = genKey("L", stage_name, str(ax.var.name))
            if key  in ctx.scheduled_axes:
                # Only for successive axes
                if len(res) > 0:
                    break
                continue
            # Only for data parallel
            if ax.iter_type != ax.DataPar:
                break
            res.append(ax)
        return res
    
    def replaceAxes(self, stage, ori_ax_names, replaced_dicts):
        res = []
        for name in ori_ax_names:
            if name in replaced_dicts.keys():
                new_name = replaced_dicts[name]
            else:
                new_name = name
            res.append(mapNametoAxis(stage, new_name))
        return res
    
    def define_com_pos(self, src_stage_name, stage, pos_type, ctx):
        key = genKey("P", stage.op.name, param_name = pos_type)
        ctx.knob_manager.define_value(key, 0, 10, 0, True)
        pos_name = stage.leaf_iter_vars[-1].var.name
        ctx.compute_poses[src_stage_name] = (stage.op.name, key)
        if stage.op.name not in ctx.compute_pos_names.keys():
            ctx.compute_pos_names[stage.op.name] = []

        if pos_name not in ctx.compute_pos_names[stage.op.name]:
            ctx.compute_pos_names[stage.op.name].append(pos_name)
        return key

    def check(self, s, stage_name, ctx):
        return NotImplementedError()

    def claim_knobs(self, s, stage_name, ctx):
        return NotImplementedError()

    def perform(self, s, stage_name, ctx):
        return NotImplementedError()

class TileAllOp(schedOp):
    def __init__(self, name):
        self.name = name
        self.tag = "None"

    def check(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        all_axes = stage.leaf_iter_vars
        spatial_, reduce_= self.findUnscheduledAxes(ctx, stage_name, all_axes)
        if len(spatial_) == 0 and len(reduce_) == 0:
            raise ValueError('No ax')
        return spatial_, reduce_
    
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## tile \n", True)
        stage = mapNametoStage(s, stage_name)
        spatial_, reduce_ = self.check(s, stage_name, ctx) 
        s_outer, s_inner, s_keys = self.tileAxes(s, stage_name, ctx, spatial_)
        r_outer, r_inner, r_keys = self.tileAxes(s, stage_name, ctx, reduce_)
        neworder = tuple(s_outer + r_outer + s_inner + r_inner)
        reorder(ctx, stage, neworder)
        sout_names = [x.var.name for x in s_outer] 
        rout_names = [x.var.name for x in r_outer] 
        # add into tile structures
        if len(sout_names) > 0:
            ctx.addSTileStucture(stage_name, sout_names, self.tag)
        if len(rout_names) > 0:
            ctx.addRTileStucture(stage_name, rout_names, self.tag)
        return s_keys, r_keys

class TileSpatialOp(TileAllOp):
    def check(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        all_axes = stage.leaf_iter_vars
        axes = self.findUnscheduledSpatialAxes(ctx, stage_name, all_axes)
        if len(axes) == 0:
            raise ValueError('No ax')
        return axes
    
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## tile spatial \n", True)
        stage = mapNametoStage(s, stage_name)
        axes = self.check(s, stage_name, ctx) 
        outer, inner, keys = self.tileAxes(s, stage_name, ctx, axes)
        neworder = tuple(outer + inner)
        reorder(ctx, stage, neworder)
        out_names = [x.var.name for x in outer] 
        # add into tile structures
        ctx.addSTileStucture(stage_name, out_names, self.tag)
        return outer, inner, keys

class generalTileOp(schedOp):
    def check(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        all_axes = stage.leaf_iter_vars
        axes = self.findUnscheduledAxes(ctx, stage_name, all_axes)
        if len(axes) == 0:
            raise ValueError('No ax')
        return axes
    
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## general tile \n", True)
        for i in range(3):
            tile_op = TileAllOp('tileAll')
            tile_op.perform(s, stage_name, ctx)


class fuseAllOp(schedOp):
    def perform(self, s, stage_name, ctx, update = True):
        stage = mapNametoStage(s, stage_name)
        axes = []
        for ax in stage.leaf_iter_vars:
            key = genKey("L", stage_name, str(ax.var.name))
            if key in ctx.scheduled_axes:
                continue
            axes.append(ax)
        fused = fuse(ctx, stage, axes, update)
        return fused

class computeAtOp(schedOp):
    def perform(self, s, stage_name, ctx):
        if stage_name not in ctx.compute_poses.keys():
            raise ValueError("Pos error")
        dst_name, knob_key = ctx.compute_poses[stage_name]
        stage = mapNametoStage(s, stage_name)
        dst_stage = mapNametoStage(s, dst_name)
        pos_id = ctx.knob_manager.get_val(knob_key)
        pos_names = ctx.compute_pos_names[dst_name]
        ax = mapNametoAxis(dst_stage, pos_names[pos_id])
        compute_at(ctx, stage, dst_stage, ax)


class InlineOp(schedOp):
    def perform(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        compute_inline(ctx, stage)
        ctx.inlined_stages.append(stage_name)

class skipOp(schedOp):
    def perform(self, s, stage_name, ctx):
        return

class startOp(schedOp):
    def genFusedAxisLength(self, s, stage_name, ctx, ax_name, map_name, coe):
        c_name, _ = ctx.compute_poses[stage_name]
        c_stage = mapNametoStage(s, c_name)
        # Pos id used for selection
        pos_key = ctx.compute_poses[stage_name][1]
        _type, _idx = map_name.split('_')
        ax_idx = int(_idx)
        if _type == 'r':
            root_ax = c_stage.op.reduce_axis[ax_idx]
        elif _type == 's':
            root_ax = c_stage.op.axis[ax_idx]
        else:
            assert 0
        root_key = genKey("L", c_name, str(root_ax.var.name))
        sub_keys = []; all_keys = []
        for ax in c_stage.leaf_iter_vars:
            key = genKey("L", c_name, str(ax.var.name))
            if 'fused' in ax.var.name:
                ori_keys = ctx.knob_manager.axis_parents[key]
                all_keys += ori_keys
            else:
                all_keys.append(key)

        for key in all_keys:
            if not root_key == ctx.knob_manager.get_axis_roots(key)[0]:
                continue
            sub_keys.append(key)
        # Generate producer's ax length
        pos_extent = ctx.knob_manager.solver.vals[pos_key].up
        length_keys = []
        for idx in range(0, pos_extent + 1):
            prod_keys = sub_keys[idx + 1 :]
            if len(prod_keys) > 1:
                length_keys.append(ctx.knob_manager.axis_parents[prod_keys[0]][0])
            elif len(prod_keys) == 1:
                # Leaf node
                length_keys.append(prod_keys[0])
            else:
                length_keys.append(1) 
        dst_key = genKey("L", stage_name, ax_name)
        ctx.knob_manager.define_value(dst_key, 1, root_ax.dom.extent.value, 1)
        if dst_key in coe:
            tmp_key = genKey("O", stage_name, ax_name, others = "tmp")
            ctx.knob_manager.define_value(tmp_key, 1, root_ax.dom.extent.value, 1)
            ctx.knob_manager.addSelect(length_keys, pos_key, tmp_key)
            ctx.knob_manager.addProd([tmp_key, coe[dst_key]], dst_key)
        else:
            ctx.knob_manager.addSelect(length_keys, pos_key, dst_key)


    def genAxesLength(self, s, stage_name, ctx):
        if not stage_name in ctx.axis_map:
            return
        # Fix axis length when on tensorcore
        coe = self.fixAxesLength(s, stage_name, ctx)
        stage = mapNametoStage(s, stage_name)
        for idx, ax in enumerate(stage.op.axis):
            self.genFusedAxisLength(s, stage_name, ctx, ax.var.name, ctx.axis_map[stage_name][idx], coe)

    def fixAxesLength(self, s, stage_name, ctx):
        return {}

    def recordaxes(self, s, stage_name, ctx):
        stage = mapNametoStage(s, stage_name)
        desc = ''
        if hasattr(stage.op, 'axis'):
            par_axes = stage.op.axis
            for idx, ax in enumerate(par_axes):
                if idx != len(par_axes) - 1:
                    desc += ax.var.name + ','
                else:
                    desc += ax.var.name
                    desc += ' = s[%s].op.axis\n'%stage_name
        if hasattr(stage.op, 'reduce_axis'):
            red_axes = stage.op.reduce_axis
            for idx, ax in enumerate(red_axes):
                if idx != len(red_axes) - 1:
                    desc += ax.var.name + ','
                else:
                    desc += ax.var.name
                    desc += ' = s[%s].op.reduce_axis\n'%stage_name
        ctx.addSchedDesc(desc)

    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n#==--------- Start schedule STAGE %s ----------==#\n"%(stage_name), True)
    #   self.recordaxes(s, stage_name, ctx)
        if stage_name not in ctx.compute_poses.keys():
            return
        compute_at_op = computeAtOp("computeAt")
        compute_at_op.perform(s, stage_name, ctx)

        # Update axes length after compute at
        self.genAxesLength(s, stage_name, ctx)

        # Print
        printAxes(s, stage_name, ctx)

class finishOp(schedOp):
    def update_compute_at_candidates(self, s, stage_name, ctx):
        pass

    def perform(self, s, stage_name, ctx):
        printAxes(s, stage_name, ctx)
        self.update_compute_at_candidates(s, stage_name, ctx)

class unrollPragmaOp(schedOp):
    def perform(self, s, stage_name, ctx):
        ctx.addSchedDesc("\n## Unroll pragma \n", True)
        stage = mapNametoStage(s, stage_name)
        unroll_key = genKey("P", stage_name, param_name = "unroll_pragma")
        ctx.knob_manager.define_value(unroll_key, 0, 5, 0, True)
        ctx.knob_manager.addRawCandidates(unroll_key, [0, 1, 2, 3, 4, 5])
        axes = stage.leaf_iter_vars
        # tile for unroll
        axes = self.findUnscheduledSpatialAxes(ctx, stage_name, axes)
        outer = []; inner = []
        for ax in axes:
            xo, xi = split(ctx, stage, ax, 1, nparts = 1)
            outer.append(xo); inner.append(xi)
            key = genKey("L", stage_name, str(xo.var.name))
            ctx.axis_anotations[key] = 'unroll'
        reorder(ctx, stage, outer + inner)
        ctx.addSTileStucture(stage_name, [x.var.name for x in outer], "unroll")
        ctx.unroll_pragma_desc[stage_name] = (outer[0].var.name, unroll_key)
        ctx.unrolled_stages.append(stage_name)



