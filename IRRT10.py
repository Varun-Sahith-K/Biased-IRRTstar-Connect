import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None
        self.reward=0

class AreaBounds:
    def __init__(self, area):
        self.xmin = float(area[0])
        self.xmax = float(area[1])
        self.ymin = float(area[2])
        self.ymax = float(area[3])

class InformedRRTStar:
    def __init__(self, start, goal, obstacle_list,
                 expand_dis=1.0, path_resolution=0.5, goal_sample_rate=10,
                 max_iter=100000):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.robot_radius = 0
        

        f1_x, f1_y = self.start.x, self.start.y
        f2_x, f2_y = self.goal.x, self.goal.y
        self.major_radius = 0.5 * math.sqrt((f1_x - f2_x)**2 + (f1_y - f2_y)**2)
        self.minor_radius = math.sqrt(abs(f1_x - f2_x)**2)
        self.center = ((f1_x + f2_x) / 2, (f1_y + f2_y) / 2)
        self.angle = math.atan2(f2_x - f1_x, f2_y - f1_y)

        a, b = self.major_radius, self.minor_radius
        x_min, x_max = self.center[0] - a, self.center[0] + a
        y_min, y_max = self.center[1] - b, self.center[1] + b
        self.play_area = ((x_min, x_max), (y_min, y_max))
        self.rand_area = ((x_min, x_max), (y_min, y_max))

        self.start_tree = []
        self.goal_tree = []

    def planning(self, animation=True):
        self.start_tree = [self.start]
        self.goal_tree = [self.goal]
        c_best = float("inf")
        last_distance_to_goal = float("inf")

        for i in range(self.max_iter):
            rnd_node = self.get_random_node(c_best)
            if not self.start_tree:
                continue
            
            rt = self.calc_dist_to_goal(rnd_node.x, rnd_node.y)
            reward = self.compute_reward(rt, last_distance_to_goal, rnd_node)
            last_distance_to_goal = rt

            if reward < 70000:
                continue

            nearest_start_to_goal = self.get_nearest_node(self.start_tree, self.goal)
            nearest_goal_to_start = self.get_nearest_node(self.goal_tree, self.start)

            if nearest_start_to_goal and self.calc_distance(nearest_start_to_goal, self.goal) <= self.expand_dis and \
               self.check_collision(nearest_start_to_goal):
                final_path = self.generate_final_path(nearest_start_to_goal, self.goal)
                return final_path

            if nearest_goal_to_start and self.calc_distance(nearest_goal_to_start, self.start) <= self.expand_dis and \
               self.check_collision(nearest_goal_to_start):
                final_path = self.generate_final_path(self.start, nearest_goal_to_start)
                return final_path

            nearest_node = self.get_nearest_node(self.start_tree, rnd_node)
            if nearest_node is None:
                continue 

            new_node = self.steer(nearest_node, rnd_node)

            if new_node and not self.check_if_outside_play_area(new_node) and self.check_collision(new_node):
                new_node.reward = reward
                self.start_tree.append(new_node)

                rt = self.calc_dist_to_goal(new_node.x, new_node.y)
                reward = self.compute_reward(rt, last_distance_to_goal, new_node)
                last_distance_to_goal = rt

                if not self.goal_tree:
                    continue

                nearest_node_goal_tree = self.get_nearest_node(self.goal_tree, new_node)
                if nearest_node_goal_tree and self.calc_distance(nearest_node_goal_tree, new_node) <= self.expand_dis:
                    new_goal_node = self.steer(nearest_node_goal_tree, new_node)
                    if new_goal_node and self.check_collision(new_goal_node):
                        self.goal_tree.append(new_goal_node)
                        if self.calc_distance(new_goal_node, self.goal) <= self.expand_dis:
                            final_path = self.generate_final_path(new_node, new_goal_node)
                            return final_path

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

        return None
    
    
    def calc_reward_ang(self, node, parent, goal):
        if parent is None:
            return 0
        
        vector1 = np.array([node.x - parent.x, node.y - parent.y])
        vector2 = np.array([goal.x - node.x, goal.y - node.y])
        
        unit_vector1 = vector1 / np.linalg.norm(vector1)
        unit_vector2 = vector2 / np.linalg.norm(vector2)
        
        dot_product = np.dot(unit_vector1, unit_vector2)
        anglerew = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        return anglerew

    def compute_reward(self, rt, rt_prev, new_node):
        
        
        ro = float('inf')
        for obstacle in self.obstacle_list:
            x1, y1, x2, y2 = obstacle[0][0], obstacle[0][1], obstacle[1][0], obstacle[1][1]
            dist = self.calc_dist_to_line_segment((new_node.x, new_node.y), (x1, y1), (x2, y2))
            ro = min(ro, dist)
        
        rc=self.robot_radius + (self.robot_radius*0.1)
        
        if ro < rc:
            Rlc = -200
        elif rc <= ro <= self.expand_dis:
            Rlc = -0.5 * (rc / ro)
        else:
            Rlc = 0

        Rlg = 180 * (rt_prev - rt) if rt_prev >= rt else -10000
        
        anglerew = self.calc_reward_ang(new_node, new_node.parent, self.goal)
        al = anglerew / np.pi
        
        if 0.1 <= al <= 0.2:
            Rla = 0
        else:
            Rla = -4 * abs(0.15 - al)  

        return Rlc + Rlg + Rla
    
    
    def calc_dist_to_line_segment(self, point, start_point, end_point):

        if self.is_point_between(point, start_point, end_point):
            return self.calc_distance(point, start_point)
        
        px, py = self.project_point_onto_line(point, start_point, end_point)
        return self.calc_distance(point, (px, py))

    def is_point_between(self, point, start_point, end_point):

        dot_product = (point[0] - start_point[0]) * (end_point[0] - start_point[0]) + (point[1] - start_point[1]) * (end_point[1] - start_point[1])
        return dot_product >= 0 and self.calc_distance(point, start_point) ** 2 <= self.calc_distance(start_point, end_point) ** 2

    def project_point_onto_line(self, point, start_point, end_point):

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        t = ((point[0] - start_point[0]) * dx + (point[1] - start_point[1]) * dy) / (dx * dx + dy * dy)
        px = start_point[0] + t * dx
        py = start_point[1] + t * dy

        return px, py
    
    def check_collision_with_radius(self, point):
        x, y = point
        for (x1, y1), (x2, y2) in self.obstacle_list:
            if (x1 - self.robot_radius < x < x2 + self.robot_radius and
                y1 - self.robot_radius < y < y2 + self.robot_radius):
                return True

            # Check corners
            corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
            for corner in corners:
                if self.calc_distance(point, corner) < self.robot_radius:
                    return True

        return False
    

    def recursive_planning_between_subgoals(self, subgoal1, subgoal2, max_recursive_iterations=3):
        path = self.planning_between_subgoals(subgoal1, subgoal2)
        best_path = path
        best_reward = self.compute_reward(path)

        for _ in range(max_recursive_iterations):
            new_path = self.planning_between_subgoals(subgoal1, subgoal2)
            new_reward = self.compute_reward(new_path)
            if new_reward > best_reward:
                best_path = new_path
                best_reward = new_reward
            else:
                break

        return best_path

    def planning_between_subgoals(self, subgoal1, subgoal2):
        self.start_tree = [subgoal1]
        self.goal_tree = [subgoal2]
        return self.planning(animation=False)

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = RRTNode(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        if extend_length > d:
            extend_length = d
            
        n_expand = math.floor(extend_length / self.path_resolution)
        if n_expand == 0:
            n_expand = 1
        
        for _ in range(n_expand):
            new_x = new_node.x + self.path_resolution * math.cos(theta)
            new_y = new_node.y + self.path_resolution * math.sin(theta)
            
            if self.check_collision_with_radius((new_x, new_y)):
                break

            new_node.x = new_x
            new_node.y = new_y
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        new_node.parent = from_node
        return new_node

    def get_random_node(self, c_best):
        if random.randint(0, 100) > self.goal_sample_rate:
            while True:
                # Shrink the ellipse by the robot radius
                a = max(0, self.major_radius - self.robot_radius)
                b = max(0, self.minor_radius - self.robot_radius)
                
                theta = random.uniform(0, 2 * math.pi)
                r = math.sqrt(random.uniform(0, 1))
                x = r * math.cos(theta) * a
                y = r * math.sin(theta) * b
                x += self.center[0]
                y += self.center[1]
                
                
                x_rotated = math.cos(2*math.pi-self.angle) * (x - self.center[0]) - math.sin(2*math.pi-self.angle) * (y - self.center[1]) + self.center[0]
                y_rotated = math.sin(2*math.pi-self.angle) * (x - self.center[0]) + math.cos(2*math.pi-self.angle) * (y - self.center[1]) + self.center[1]
                
                
                if (self.play_area[0][0] + self.robot_radius <= x_rotated <= self.play_area[0][1] - self.robot_radius and 
                    self.play_area[1][0] + self.robot_radius <= y_rotated <= self.play_area[1][1] - self.robot_radius and
                    not self.check_collision_with_radius((x_rotated, y_rotated))):
                    rnd = RRTNode(x_rotated, y_rotated)
                    break
        else:
            rnd = RRTNode(self.goal.x, self.goal.y)
        return rnd

    def generate_final_path(self, start_node, goal_node):
        path = [[self.goal.x, self.goal.y]]
        rewards = []
        node = start_node
        while node.parent is not None:
            path.append([node.x, node.y])
            rewards.append(node.reward)
            node = node.parent
        path.append([node.x, node.y])
        rewards.append(node.reward)
        node = goal_node
        while node.parent is not None:
            path.insert(0, [node.x, node.y])
            rewards.insert(0, node.reward)
            node = node.parent
        path.insert(0, [node.x, node.y])
        rewards.insert(0, node.reward)
        
        # Smooth the path
        smoothed_path = self.smooth_path(path)
        return smoothed_path, rewards
    
    def smooth_path(self, path):
        smoothed_path = path.copy()
        i = 0
        while i < len(smoothed_path) - 2:
            current_node = smoothed_path[i]
            j = len(smoothed_path) - 1
            while j > i + 1:
                check_node = smoothed_path[j]
                if not self.check_collision_between(current_node, check_node):
                    # Remove all nodes between i and j
                    del smoothed_path[i+1:j]
                    j = i + 1  # Move j to the next node after i
                else:
                    j -= 1
            i += 1
        return smoothed_path

    def check_collision_between(self, node1, node2):
        # Check if the line between node1 and node2 collides with any obstacle
        x1, y1 = node1
        x2, y2 = node2
        for step in range(int(self.calc_distance((x1, y1), (x2, y2)) / self.path_resolution)):
            t = step / (self.calc_distance((x1, y1), (x2, y2)) / self.path_resolution)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if self.check_collision_with_radius((x, y)):
                return True
        return False
    

    def calc_dist_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        return math.hypot(dx, dy)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        return distance, angle

    def calc_distance(self, node1, node2):
        if isinstance(node1, tuple) and isinstance(node2, tuple):
            return math.hypot(node1[0] - node2[0], node1[1] - node2[1])
        else:
            return math.hypot(node1.x - node2.x, node1.y - node2.y)

    def get_nearest_node(self, start_tree, rnd_node):
        dlist = [self.calc_distance(node, rnd_node) for node in start_tree]
        min_index = dlist.index(min(dlist))
        
        nearest_node = start_tree[min_index]
        if not self.check_if_outside_play_area(nearest_node):
            return nearest_node
        else:
            return None


    def check_collision(self, node):
        if not node.path_x or not node.path_y:
            return False

        for i in range(len(node.path_x)):
            if self.check_collision_with_radius((node.path_x[i], node.path_y[i])):
                return False

        return True

    def check_if_outside_play_area(self, node):
        if self.play_area is None:
            return False

        return not (self.play_area[0][0] <= node.x <= self.play_area[0][1] and
                    self.play_area[1][0] <= node.y <= self.play_area[1][1])

    def calculate_angle(self):
        dx = self.start.x - self.goal.x
        dy = self.start.y - self.goal.y
        angle = math.atan2(dx, dy)
        return angle


    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        angle = self.calculate_angle()
        ellipse=mpatches.Ellipse(self.center, 2*self.major_radius, 2*self.minor_radius, angle=-np.degrees(angle), fill=False)
        plt.gca().add_patch(ellipse)

        for node in self.start_tree:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "black")

        for node in self.goal_tree:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-r")
                

        for (x1, y1), (x2, y2) in self.obstacle_list:
            plt.fill([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='b')

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            
            for x, y in path:
                circle = plt.Circle((x, y), self.robot_radius, color='g', fill=False, alpha=0.5)
                plt.gca().add_artist(circle)
                
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def main():
    print("Start Informed RRT* planning")

    '''obstacle_list_1 = [
        ((275, 0), (325, 200)), # rectangle defined by bottom-left and top-right corners
        ((275, 450), (325, 1000)),
        ((675, 800), (725, 1000)),
        ((675, 0), (725, 650))
    ]'''
    
    obstacle_list_1 = [
        ((50,25), (125,100)),
        ((225,50), (300,125)),
        ((325,100), (400,175)),
        ((175,175), (250,250)),
        ((50,300), (125,375)),
        ((50,500), (125,575)),
        ((50,725), (125,800)),
        ((150,850), (225,925)),
        ((150,700), (225,775)),
        ((275,575), (350,650)),
        ((250,450), (325,525)),
        ((250,775), (325,850)),
        ((275,325), (350,400)),
        ((350,850), (425,925)),
        ((600,325), (675,400)),
        ((375,500), (450,575)),
        ((425,375), (500,450)),
        ((375,225), (450,300)),
        ((475,150), (550,225)),
        ((500,525), (575,600)),
        ((600,100), (675,175)),
        ((600,725), (675,800)),
        ((600,850), (675,925)),
        ((750,25), (825,100)),
        ((750,850), (825,925)),
        ((400,650), (475,725)),
        ((475,800), (550,875)),
        ((550,650), (625,725)),
        ((700,450), (775,525)),
        ((700,200), (775,275)),
        ((650,600), (725,675)),
        ((800,325), (875,400)),
        ((900,200), (975,275)),
        ((625,800), (700,875)),
        ((775,725), (850,800)),
        ((900,900), (975,975)),
        ((825,475), (900,550)),
        ((825,675), (900,750)),
        ((900,600), (975,675)),
        ((900,400), (975,475))
        
    ]

    rrt_star = InformedRRTStar(
        start=[300, 700],
        goal=[800, 200],
        obstacle_list=obstacle_list_1)

    res = rrt_star.planning(animation=False)
        
    if res is None:
        print("No path found!")
    else:
        path, rewards = res
        print("Found path!")
        print("Reward values for nodes in the final path:")
        for i, reward in enumerate(rewards):
            print(f"Node {i}: Reward = {reward}")
        rrt_star.draw_graph(path=path)


if __name__ == '__main__':
    main()
     
    
  