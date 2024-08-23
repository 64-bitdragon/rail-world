import amulet
from amulet.api.block import Block
from datetime import datetime
from distutils.dir_util import copy_tree
import numpy as np
import matplotlib.pyplot as plt
from amulet_nbt import StringTag
import pygad

"""
eurgh so i'm not sure this is going to work
as the boundaries between different sections will be hard to quantify looking at individual blocks
i.e. we know we have a valley coming up so we want to bridge it
vs just a random hill downwards
in which case the track should just go downhill
so instead we want to get a long list of heights
that covers the entire track length
then we can apply some analyzers
i.e. down+up == bridge
big_up+big_down == tunnel
so what would these analyzers look like?
down+up == bridge
    need logic for gap length vs drop
    i.e. small bridges for < 16 blocks
    and larger ones up to 64 blocks, if the depth is also high?
    but what if the end height doesn't match the start height?
    i.e. down5 + flat5 + up10
    then we probably want to either go into a tunnel if the up is too big
    but what if up is a lot less than the down?
    then we probably want to extend the bridge until we find a good enough up?
how about water?
    that should be easier right?
    just treat it as flat
    but bridge it maybe?
    that could be fun
    we could always check the max height of both shores
    and set the bridge height to that
    then we will want to increase the height up to that before hitting the water
    unless we are going straight into a tunnel
    or straight from a tunnel?
ok, so might be easier to classify the slopes as cliffs vs non cliffs
    but we don't want to just tunnel forever
    so it's not just about cliffs
    it's about mountains
ok,
    so it seems like there are multiple options
    i.e. we could tunnel under
        or we could slope it
        or do a mixture
    so we could try both
        and get a score for both
        and choose the one that minimizes the score
    but for scenarios that we want to partially slope and then tunnel
    maybe we have a mountain range, so we want to get up to it's base level
    and then tunnel through any really big bits
    in that case we want still want our mountain detector
    so maybe the whole scoring system could still work, but only for certain situations
so what are we going to do?
    get the heights for the whole thing
    detect mountains
    detect valleys
    detect water
    so to do all this, we probably want to use a rolling average
        and if we are going up more than one block per block, then we figure out winding tracks
    any time we are tunneling
        ideally we want something flat
        so we could choose the highest of the entrance and exit
            and use that
            then we would have to re-connect the heights
            so maybe use winding for that?
            that could work very nicely
    any time we are going over water
        choose the highest of the two shores
        unless one shore has a mountain that we are tunneling
        then choose the other
        and if they are both mountains, we'll figure something
    ok, so we are using rolling average to detect mountains and valleys
        but whenever we are not in a mountain or valley, we can use the flat generator
        which can follow the ground level
        but we want to set the heights here too
        rather than at a block by block basis
        because if we want to go up a steep incline
        then we will have to start earlier
        and maybe render this differently as well
        with support structures
            well ye, any time we aren't rendering on the ground
        so for 'flat' sections
        we try to follow the ground level as much as possible
            but not always
            not sure rolling average is going to help us here too much
        what are the cases?
            need to go up a steep incline
            so need a bridge up
            but it shouldn't just bridge any time the height change is > 1
            it should zoom out a bit, find the highest point, and bridge up to that
                skipping over 'local high points'
            so we could extend out a vector at 45 degrees
                and skip over the local high points
            not sure that will work
            so what else
                split the remaining flat ground into 'flat' and 'hilly'
                these will alternate
                connect two 'flat' areas with a 45 degree bridge
                and make sure it skips over any 'local high points' in the hilly area
            that could work!
"""

"""
    so in terms of code flow
        we get the list of heights
        remove leaves
        remove trees
        do a rolling average on the heights
        detect mountains, valleys, flat, sea
        then adjust something?
        still feel like adjusting and joining is going to be tough
        
"""

"""Path to existing minecraft world, don't put a / on the end!"""
level_path = "C:/Users/Home/AppData/Roaming/.minecraft/saves/railWorld"
game_version = ("java", (1, 20, 0))

dry_run = False or __name__ != '__main__'

if not dry_run:
    # Makes a copy of the existing level, appending the current time to the end of the name
    time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    copy_tree(level_path, level_path + time)

    # Then opens it as an Amulet level
    # minecraft_level = amulet.load_level(level_path)
    minecraft_level = amulet.load_level(level_path + time)
    minecraft_level.level_wrapper.level_name += time
else:
    minecraft_level = amulet.load_level(level_path)


class ChunkWrapper:
    def __init__(self, cx):
        self.cx = cx
        self.chunk = minecraft_level.get_chunk(cx, 0, "minecraft:overworld")
        self.ground_heights = self.chunk.misc['height_mapC']['WORLD_SURFACE'][8]

    def has_room_for_station(self):
        chunk_heights = self.chunk.misc['height_mapC']['WORLD_SURFACE']

        for i in range(0, 16):
            for j in range(0, 16):
                block_name = self.chunk.get_block(i, chunk_heights[i, j] - 1, j).base_name
                if block_name == 'water':
                    return False

        chunk_heights = np.reshape(chunk_heights, [256])
        if np.std(chunk_heights) < 1:
            return True

        return False

    def remove_trees_from_ground_heights(self):
        for i in range(16):
            while True:
                block_name = self.chunk.get_block(i, self.ground_heights[i] - 1, 8).base_name

                if block_name in ['plant',
                                  'log',
                                  'seagrass',
                                  'tall_seagrass',
                                  'sugar_cane',
                                  'double_plant',
                                  'leaves',
                                  'vine',
                                  'air']:
                    self.ground_heights[i] -= 1
                else:
                    break

    def get_block_name_at_position(self, x, ground_level_offset):
        return self.chunk.get_block(x, self.ground_heights[x] + ground_level_offset, 8).base_name

    def flush(self):
        minecraft_level.put_chunk(self.chunk, "minecraft:overworld")

    def place_rail_at_height(self, cx, height, shape):
        self.chunk.set_block(
            cx,
            height,
            8,
            Block("minecraft", "rail", {"shape": StringTag(shape)}))

    def set_block(self, cx, height, block_name, z_offset=0, state=None):
        self.chunk.set_block(
            cx,
            height,
            8 + z_offset,
            Block("minecraft", block_name, state))

    def place_powered_rail_at_height(self, cx, height, shape):
        self.chunk.set_block(
            cx,
            height,
            8,
            Block("minecraft", "powered_rail", {"shape": StringTag(shape), "powered": StringTag("true")}))

    def place_rails_at_heights(self, heights):
        for i in range(16):
            self.chunk.set_block(
                i,
                heights[i],
                8,
                Block("minecraft", "rail", {"shape": StringTag("east_west")}))


class HeightManager:
    TUNNEL_START = 0
    FLAT_START = 1
    BRIDGE_START = 2
    END = 3

    key_points = {}

    def __init__(self, ground_heights):
        self.ground_heights = ground_heights
        self.length = len(ground_heights)
        self.heights = [x for x in ground_heights]

        self.key_points[0] = self.FLAT_START
        self.key_points[self.length - 1] = self.END

        self.differential = np.zeros([len(self.ground_heights)])

    def get_key_points_of_type(self, type):
        """
        :param type: TUNNEL_START, FLAT_START
        :return: a list of tuples of form (start, end)
        """
        sorted_keys = sorted(self.key_points.keys())
        sorted_keys.append(self.length)

        for i in range(0, len(sorted_keys) - 1):
            if self.key_points[sorted_keys[i]] == type:
                yield sorted_keys[i], sorted_keys[i + 1]

    def add_keypoint(self, keypoint_type, start, length):
        self.key_points[start] = keypoint_type
        self.key_points[start + length] = self.key_points[self.get_first_keypoint_before(start)]

    def get_first_keypoint_before(self, point):
        sorted_keys = sorted(self.key_points.keys())

        previous = 0
        for key in sorted_keys:
            if key >= point:
                return previous

            previous = key

        return previous

    def get_first_keypoint_after(self, point):
        sorted_keys = sorted(self.key_points.keys())

        for key in sorted_keys:
            if key > point:
                return key

        return self.length

    def render_ground(self):
        plt.plot([x for x in range(len(self.ground_heights))], self.ground_heights)
        plt.show()

    def plot_heights(self):
        fig, ax = plt.subplots()

        # set xticks & yticks
        ax.set(xticks=range(0, 10, 1), yticks=range(75, 85, 1))
        ax.set_aspect('equal')
        # draw grid
        for loc in range(0, 10, 1):
            ax.axvline(loc, alpha=0.5, color='#b0b0b0', linestyle='-', linewidth=0.8)

        for loc in range(75, 85, 1):
            ax.axhline(loc, alpha=0.5, color='#b0b0b0', linestyle='-', linewidth=0.8)

        x_axis = [x for x in range(10)]

        plt.plot(x_axis, self.ground_heights[20:30])
        #plt.plot(x_axis, self.heights)
        #plt.legend(['ground heights', 'heights', 'diff'], loc='upper left')
        plt.show()


class MountainHelper:
    def __init__(self, height_manager: HeightManager):
        self.height_manager = height_manager

    def get_rolling_average(self):
        window = 80
        rolling = np.convolve(self.height_manager.ground_heights, np.ones(window) / window, mode='same')

        for i in range(window // 2):
            rolling[i] = self.height_manager.ground_heights[i]
            rolling[-i] = self.height_manager.ground_heights[-i]

        return rolling

    def remove_mountains(self,):
        average = self.get_rolling_average()

        # go through each point
        i = 0
        while i < self.height_manager.length:
            #print(f'i: {i}, ground: {height_manager.heights[i]}, average: {average[i]}')
            # and check the difference between the real height and the average height
            diff = self.height_manager.heights[i] - average[i]
            #print(f'diff: {diff}')

            if diff > 5:
                # if the difference is great enough
                # then we switch to a tunnel

                tunnel_height = average[i]
                tunnel_start = i
                tunnel_end = i

                # we go backwards to find the entrance
                a = i
                while a >= 0:
                    if self.height_manager.heights[a] < average[a] or self.height_manager.heights[a] <= tunnel_height:
                        tunnel_start = a
                        break

                    a -= 1

                # then go forwards to find the exit
                a = i
                while a < self.height_manager.length:
                    if self.height_manager.heights[a] < average[a] or self.height_manager.heights[a] <= tunnel_height:
                        tunnel_end = a
                        break

                    a += 1

                # reset the tunnel_height to match either the start or end of the tunnel
                tunnel_height = max(self.height_manager.heights[tunnel_start], self.height_manager.heights[tunnel_end])

                # and finally make the tunnel
                #print(f'start: {tunnel_start}, stop: {tunnel_end}')
                for k in range(tunnel_start, tunnel_end):
                    self.height_manager.heights[k] = tunnel_height

                self.height_manager.add_keypoint(self.height_manager.TUNNEL_START, tunnel_start, tunnel_end - tunnel_start)
                i = tunnel_end

            i += 1

    def plot_heights(self):
        fig, ax = plt.subplots()

        x_axis = [x for x in range(len(self.height_manager.heights))]

        plt.plot(x_axis, self.height_manager.ground_heights)
        rolling_average = self.get_rolling_average()
        plt.plot(x_axis, rolling_average)
        plt.plot(x_axis, self.height_manager.heights)
        #plt.legend(['ground heights', 'heights', 'diff'], loc='upper left')
        plt.show()

    def render_to_minecraft(self, chunks):
        key_points = self.height_manager.get_key_points_of_type(self.height_manager.TUNNEL_START)

        for start, end in key_points:
            for i in range(start, end):
                chunk_index = int(i / 16)
                cx = i % 16
                chunk = chunks[chunk_index]

                for j in [-1, 0, 1]:
                    chunk.set_block(cx, self.height_manager.heights[i] - 1, 'stone', z_offset=j)
                    chunk.set_block(cx, self.height_manager.heights[i], 'air', z_offset=j)
                    chunk.set_block(cx, self.height_manager.heights[i] + 1, "air", z_offset=j)

                if (i % 8) == 0:
                    chunk.set_block(cx, self.height_manager.heights[i], 'torch', z_offset=-1)
                    chunk.set_block(cx, self.height_manager.heights[i], 'torch', z_offset=1)

            # render the tunnel starts and ends
            # for k in [start, end]:
            #     chunk_index = int(k / 16)
            #     cx = k % 16
            #     chunk = chunks[chunk_index]
            #     for i in [-1, 1]:
            #         for j in range(0, 2):
            #             chunk.set_block(cx, self.height_manager.heights[k] + j, 'oak_fence', z_offset=i)
            #
            #     for i in range(-1, 2):
            #         chunk.set_block(cx, self.height_manager.heights[k] + 2, 'oak_planks', z_offset=i)


class ValleyHelper:
    def __init__(self, height_manager: HeightManager):
        self.height_manager = height_manager

    def get_rolling_average(self):
        window = 64
        rolling = np.convolve(self.height_manager.heights, np.ones(window) / window, mode='same')

        for i in range(window // 2):
            rolling[i] = self.height_manager.heights[i]
            rolling[-i] = self.height_manager.heights[-i]

        return rolling

    def remove_valleys(self, ):
        average = self.get_rolling_average()

        # go through each point
        i = 0
        while i < self.height_manager.length:
            #print(f'i: {i}, ground: {self.height_manager.heights[i]}, average: {average[i]}')
            # and check the difference between the real height and the average height
            diff = self.height_manager.heights[i] - average[i]
            #print(f'diff: {diff}')

            if diff < -3:
                # if the difference is great enough
                # then we switch to a bridge
                #print('creating bridge')

                bridge_height = average[i]
                bridge_start = i
                bridge_end = i

                # we go backwards to find the entrance
                a = i - 1
                while a >= 0:
                    if self.height_manager.heights[a] >= bridge_height:
                        bridge_start = a
                        break

                    a -= 1

                # then go forwards to find the exit
                a = i + 1
                while a < self.height_manager.length:
                    if self.height_manager.heights[a] >= bridge_height:
                        bridge_end = a
                        break

                    a += 1

                if bridge_end == i:
                    # couldn't find sensible end, returning
                    i += 1
                    continue

                #print('bridge length: ', (bridge_end - bridge_start))
                if bridge_end - bridge_start > 100:
                    i += 1
                    continue

                # reset the bridge_height to match either the start or end of the bridge
                tunnel_height = max(self.height_manager.heights[bridge_start], self.height_manager.heights[bridge_end])

                # and finally make the bridge
                #print(f'start: {tunnel_start}, stop: {tunnel_end}')
                for k in range(bridge_start, bridge_end):
                    self.height_manager.heights[k] = tunnel_height

                self.height_manager.add_keypoint(self.height_manager.BRIDGE_START, bridge_start, bridge_end - bridge_start)
                i = bridge_end

            i += 1

    def render_to_minecraft(self, chunks):
        key_points = self.height_manager.get_key_points_of_type(self.height_manager.BRIDGE_START)

        for start, end in key_points:
            for i in range(start, end):
                chunk_index = int(i / 16)
                cx = i % 16
                chunk = chunks[chunk_index]

                for h in range(self.height_manager.ground_heights[i], self.height_manager.heights[i]):
                    chunk.set_block(cx, h, 'oak_fence')

                for j in [-1, 1]:
                    chunk.set_block(cx, self.height_manager.heights[i] - 1, 'oak_planks', z_offset=j)

                    chunk.set_block(cx, self.height_manager.heights[i], 'oak_fence', z_offset=j,
                                    state={'east': StringTag('true'), 'west': StringTag('true')})

                    chunk.set_block(cx, self.height_manager.heights[i] + 1, "air", z_offset=j)

                chunk.set_block(cx, self.height_manager.heights[i] - 1, 'oak_planks')
                chunk.set_block(cx, self.height_manager.heights[i], 'air')
                chunk.set_block(cx, self.height_manager.heights[i] + 1, "air")

    def plot_heights(self):
        x_axis = [x for x in range(len(self.height_manager.ground_heights))]

        average = self.get_rolling_average()

        plt.plot(x_axis, self.height_manager.ground_heights)
        plt.plot(x_axis, average)
        plt.plot(x_axis, self.height_manager.heights)

        plt.legend(['ground heights', 'average', 'heights'], loc='upper left')
        plt.show()


class HillHelper:
    def __init__(self, height_manager: HeightManager):
        self.height_manager = height_manager

    def adjust_all_hills(self, height_manager: HeightManager):
        height_manager.heights = self.adjust_hill_segment(height_manager.heights)
        
        # for key_point in height_manager.key_points:
        #     if height_manager.key_points[key_point] == height_manager.FLAT_START:
        #         start = key_point
        #         end = height_manager.get_first_keypoint_after(start)
        #
        #         heights = height_manager.heights[start:end + 1]
        #         height_manager.heights[start:end + 1] = self.adjust_hill_segment(heights)

    def adjust_hill_segment(self, heights):
        new_heights = []

        previous = heights[0]
        new_heights.append(previous)

        i = 1
        while i < len(heights):
            current = heights[i]
            diff = current - previous

            if diff > 1:
                new_heights.append(current)

                # start adjusting heights going backwards to gradually slope upwards
                a = i
                while a > 0:
                    if new_heights[a - 1] >= new_heights[a] - 1:
                        break

                    new_heights[a - 1] = new_heights[a] - 1
                    a -= 1
            elif diff < -1:
                # start adjusting heights going forwards to gradually slope downwards

                while i < len(heights):
                    previous -= 1
                    new_heights.append(previous)
                    if heights[i] <= previous:
                        current = previous
                        break

                    i += 1
            else:
                new_heights.append(current)

            previous = current
            i += 1

        # handle cases where we dip down 1 block, like 1-0-1
        # just flatten it out
        for i in range(len(new_heights) - 2):
            if new_heights[i] - 1 == new_heights[i + 1] and new_heights[i] == new_heights[i + 2]:
                new_heights[i + 1] = new_heights[i]

        return new_heights

    def render_to_minecraft(self, chunks):
        key_points = self.height_manager.get_key_points_of_type(self.height_manager.FLAT_START)

        for start, end in key_points:
            for i in range(start, end):
                chunk_index = int(i / 16)
                cx = i % 16
                chunk = chunks[chunk_index]

                needs_fence = self.height_manager.ground_heights[i] != self.height_manager.heights[i] or chunk.get_block_name_at_position(cx, -1) == 'water'

                for h in range(self.height_manager.ground_heights[i], self.height_manager.heights[i]):
                    chunk.set_block(cx, h, 'oak_fence')

                for j in [-1, 1]:
                    chunk.set_block(cx, self.height_manager.heights[i] - 1, 'oak_planks', z_offset=j)

                    if needs_fence:
                        chunk.set_block(cx, self.height_manager.heights[i], 'oak_fence', z_offset=j,
                                        state={'east': StringTag('true'), 'west': StringTag('true')})
                    else:
                        chunk.set_block(cx, self.height_manager.heights[i], 'air', z_offset=j)

                    chunk.set_block(cx, self.height_manager.heights[i] + 1, "air", z_offset=j)

                chunk.set_block(cx, self.height_manager.heights[i] - 1, 'oak_planks')
                chunk.set_block(cx, self.height_manager.heights[i], 'air')
                chunk.set_block(cx, self.height_manager.heights[i] + 1, "air")

                is_height_change = self.height_manager.heights[i] - self.height_manager.heights[i + 1] == -1
                if not is_height_change and i > 0:
                    is_height_change = self.height_manager.heights[i] - self.height_manager.heights[i - 1] == -1

                if is_height_change:
                    chunk.set_block(cx, self.height_manager.heights[i], 'oak_fence', z_offset=-1,
                                    state={'east': StringTag('true'), 'west': StringTag('true')})
                    chunk.set_block(cx, self.height_manager.heights[i], 'oak_fence', z_offset=1,
                                    state={'east': StringTag('true'), 'west': StringTag('true')})
                    chunk.set_block(cx, self.height_manager.heights[i] + 1, 'redstone_torch', z_offset=-1)


class ObstacleHelper:
    def __init__(self, height_manager: HeightManager):
        self.height_manager = height_manager

    def remove_obstacles(self, chunks):
        for i in range(0, self.height_manager.length):
            chunk_index = int(i / 16)
            cx = i % 16
            chunk = chunks[chunk_index]

            block = chunk.get_block_name_at_position(cx, -1)
            #print(block)


class TrackRenderer:
    def __init__(self, height_manager: HeightManager):
        self.height_manager = height_manager

    def clean_upper_layer(self, chunks):
        for i in range(0, self.height_manager.length):
            chunk_index = int(i / 16)
            cx = i % 16
            chunk = chunks[chunk_index]
            chunk.set_block(cx, self.height_manager.heights[i] + 1, "air")

    def render_track(self, chunks):
        for i in range(0, self.height_manager.length - 1):
            chunk_index = int(i / 16)
            cx = i % 16
            chunk = chunks[chunk_index]

            diff = self.height_manager.heights[i] - self.height_manager.heights[i + 1]

            if  self.height_manager.heights[i] - self.height_manager.heights[i + 1] == -1:
                chunk.place_powered_rail_at_height(cx, self.height_manager.heights[i], 'ascending_east')
            elif self.height_manager.heights[i] - self.height_manager.heights[i - 1] == -1:
                chunk.place_powered_rail_at_height(cx, self.height_manager.heights[i], 'ascending_west')
            else:
                chunk.place_rail_at_height(cx, self.height_manager.heights[i], 'east_west')

def create_station(chunk: ChunkWrapper):
    for i in range(16):
        chunk.set_block(i, chunk.ground_heights[0], 'stone_slab', 1)
        chunk.set_block(i, chunk.ground_heights[0], 'stone_slab', 2)

def main():
    chunks = [ChunkWrapper(x) for x in range(60)]
    for chunk in chunks:
        chunk.remove_trees_from_ground_heights()

    all_heights = np.concatenate([c.ground_heights for c in chunks])
    height_manager = HeightManager(all_heights)

    mountain_helper = MountainHelper(height_manager)
    hill_helper = HillHelper(height_manager)
    obstacle_helper = ObstacleHelper(height_manager)
    valley_helper = ValleyHelper(height_manager)

    # obstacle_helper.remove_obstacles(chunks)
    mountain_helper.remove_mountains()
    valley_helper.remove_valleys()
    hill_helper.adjust_all_hills(height_manager)

    mountain_helper.render_to_minecraft(chunks)
    valley_helper.render_to_minecraft(chunks)
    hill_helper.render_to_minecraft(chunks)
    track_renderer = TrackRenderer(height_manager)
    track_renderer.clean_upper_layer(chunks)
    track_renderer.render_track(chunks)

    for chunk in chunks:
        if chunk.has_room_for_station():
            create_station(chunk)

    # mountain_helper.plot_heights()
    #height_manager.plot_heights()

    valley_helper.plot_heights()

    if not dry_run:
        for chunk in chunks:
            chunk.flush()

        minecraft_level.save()
        minecraft_level.close()


if __name__ == '__main__':
    main()
