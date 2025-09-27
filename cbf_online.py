import rclpy
from rclpy.node import Node
import numpy as np
import cvxpy as cp      
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
import sympy as sp
from sympy.utilities.lambdify import lambdify
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

 
Xgoal = [4, 0]
gamma = 1
T = 80
dt = 0.02

time_steps = int(np.ceil(T / dt))

class Circle:
    def __init__(self, origin=None, radius=None):
        self.origin = origin  # origin should be a tuple (x, y)
        self.radius = radius

    def closest_point(self, x0, y0):
        ox, oy = self.origin
        r = self.radius
        dx = x0 - ox
        dy = y0 - oy
        norm = np.hypot(dx, dy)
        if norm == 0:
            # Special case: point is exactly at center
            return (ox + r, oy, 0,0)
        closest_x = ox + r * dx / norm
        closest_y = oy + r * dy / norm
        theta = np.arctan2(dy, dx)
        return (closest_x, closest_y, norm, theta)

    

class CBF(Node):
    def __init__(self):
        super().__init__('cbf_node')
        self.distance_to_g_ = 100
        self.obstacle_bound_ = 0.2
        self.obstacle_theta_bound_ = 0.4
        self.detection_range_ = 1
        self.max_obstacles_ = 1
        self.not_created_yet = 1
        self.vert_obs_ = Circle()
        self.u_history_ = np.zeros((2, 3))
        self.xt_ = np.zeros((3, time_steps))
        self.ut_ = np.zeros((2, time_steps))
        self.i_ = 0
        self.subscription_odom_ = self.create_subscription(Odometry, '/odom', self.pose_callback, 10) 
        self.subscription_lazer_ = self.create_subscription(LaserScan, '/scan', self.laser_scan_callback, 10) 
        self.publisher_cmd_vel_ = self.create_publisher(Twist, '/cmd_vel', 10)
        #self.publisher_path_ = self.create_publisher(Path, '/trajectory', 10)

        
        
    def pose_callback(self, curent_pose:Odometry):
        self.xt_[0,self.i_+1] = curent_pose.pose.pose.position.x
        self.xt_[1,self.i_+1] = curent_pose.pose.pose.position.y
        q = [curent_pose.pose.pose.orientation.x, curent_pose.pose.pose.orientation.y, curent_pose.pose.pose.orientation.z, curent_pose.pose.pose.orientation.w]        
        r = R.from_quat(q)
        euler_rad = r.as_euler('xyz', degrees=False)
        self.xt_[2,self.i_+1] = euler_rad[2]
        

    def laser_scan_callback(self, laser_scan_data:LaserScan):
        #obs_dis = min(laser_scan_data.ranges)
        #min_index = laser_scan_data.ranges.index(min(laser_scan_data.ranges))
        #angle_to_obs = laser_scan_data.angle_min + min_index * laser_scan_data.angle_increment
        vel_msg = Twist()
        if self.distance_to_g_ > 0.2:
            self.ut_[:,self.i_+1] = self.cbf(laser_scan_data, Xgoal, self.xt_[:,self.i_+1], gamma)
            vel_msg.linear.x = self.ut_[0,self.i_+1]
            vel_msg.angular.z = self.ut_[1,self.i_+1]
            self.publisher_cmd_vel_.publish(vel_msg)
            self.i_ += 1
            #self.path_pub_callback()
        else:
            self.xt_ = self.xt_[:, 1:self.i_]
            self.ut_ = self.ut_[:, 0:self.i_]
            vel_msg.linear.x = 0.0
            vel_msg.angular.z = 0.0
            self.publisher_cmd_vel_.publish(vel_msg)   
            self.ploting_path()
            #self.path_pub_callback()
        
    def path_pub_callback(self):
        path = Path()
        path.header.frame_id = 'map'
        for x, y in zip(self.xt_[0, 1:self.i_], self.xt_[1, 1:self.i_]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        now = self.get_clock().now().to_msg()
        path.header.stamp = now
        self.publisher_path_.publish(path)
        self.get_logger().info('Publishing trajectory...')


    def cbf(self, laser_scan_data:LaserScan, Xgoal, xt, gamma):
        H = np.eye(2)
        x, y, theta = sp.symbols('x y theta')
        X  = sp.Matrix([x, y, theta])
        U = cp.Variable(2)
        g_symbolic = sp.Matrix([[sp.cos(theta), 0],[sp.sin(theta), 0],[0, 1]])
        angles_scan = np.arange(laser_scan_data.angle_min, laser_scan_data.angle_max + laser_scan_data.angle_increment, laser_scan_data.angle_increment)
        
        ranges = np.array(laser_scan_data.ranges)
        valid_idx = ranges <= self.detection_range_
        the_near_range = np.array(ranges[valid_idx])
        the_near_angles_scan = np.array(angles_scan[valid_idx])
        the_near_x_obs = np.column_stack((the_near_range, the_near_angles_scan))
        
        constraints = []
        if the_near_x_obs.size != 0:
            chunks = np.array_split(the_near_x_obs, self.max_obstacles_)
            the_near_x_obs = np.array([chunk[np.argmin(chunk[:, 0])] for chunk in chunks])

            CBF = []
            Lg_CBF = []
            
            for obs in the_near_x_obs:
                thetao = self.xt_[2,self.i_] + obs[1]
                xo = self.xt_[0,self.i_] + obs[0] * sp.cos(thetao)
                yo = self.xt_[1,self.i_] + obs[0] * sp.sin(thetao)
                alpha = 1.25-np.abs(np.abs(obs[1])-laser_scan_data.angle_max)/laser_scan_data.angle_max 
                cbf_symbolic = [alpha*((x - xo)**2 + (y - yo)**2) - self.obstacle_bound_**2, alpha * obs[0]* sp.tan(theta-thetao)**2 - self.obstacle_theta_bound_]
                dx_cbf_symbolic = sp.Matrix([cbf_symbolic]).jacobian(X) 
                cbf = lambdify(np.array(X.T), cbf_symbolic, 'numpy')
                lg_cbf = lambdify(np.array(X.T), dx_cbf_symbolic * g_symbolic , 'numpy')   
                CBF.append(cbf)
                Lg_CBF.append(lg_cbf)
            B = np.array(CBF[0](xt))
            lg_B = np.array(Lg_CBF[0](xt))       
            for i in range(1,self.max_obstacles_):
                B = np.vstack((B, CBF[i](xt)))
                lg_B = np.vstack((lg_B, Lg_CBF[i](xt)))
          
            A = -lg_B
            b = gamma * B.reshape(1,2*self.max_obstacles_ )
            constraints = [A @ U <= b]
        
        self.get_logger().info(str(self.detect_dense_obstacle_cluster(laser_scan_data)))
        if self.detect_dense_obstacle_cluster(laser_scan_data) or not(self.not_created_yet):
            if self.not_created_yet:
                self.vert_obs_ = self.create_vertual_obstacl(laser_scan_data)
                self.get_logger().info('(origin_x,origin_y, r): ('+str(self.vert_obs_.origin[0])+', '+str(self.vert_obs_.origin[1])+', '+str(self.vert_obs_.radius)+')')
                self.not_created_yet = 0
            else:
                x_obs, y_obs, d_obs, theta_obs= self.vert_obs_.closest_point(self.xt_[0,self.i_], self.xt_[1,self.i_])
                self.get_logger().info('(x_obs,y_obs): ('+str(x_obs)+', '+str(y_obs)+')')
                cbf_symbolic_s = [(x-x_obs)**2+(y-y_obs)**2 - self.obstacle_bound_**2, 0.4 * d_obs * sp.tan(theta-theta_obs)**2 - self.obstacle_theta_bound_]
                dx_cbf_symbolic_s = sp.Matrix([cbf_symbolic_s]).jacobian(X) 
                cbf_s = lambdify(np.array(X.T), cbf_symbolic_s, 'numpy')
                lg_cbf_s = lambdify(np.array(X.T), dx_cbf_symbolic_s * g_symbolic , 'numpy')

                
                # Constraints: A u <= b
                B_s = cbf_s(xt)
                lg_B_s = lg_cbf_s(xt)
                A_s = -lg_B_s
                b_s = gamma * B_s
                constraints += [A_s @ U <= b_s]
        e_x = Xgoal[0] - xt[0]
        e_y = Xgoal[1] - xt[1]
        self.distance_to_g_ = np.sqrt(e_x**2 + e_y**2)
        theta_d = np.arctan2(e_y, e_x)
        theta_to_g = theta_d - xt[2]
        if theta_to_g < -3.14:
            theta_to_g = theta_to_g + 6.28
        elif theta_to_g > 3.14:
            theta_to_g = theta_to_g - 6.28

        K_att = 1
        obj_function = np.array([[K_att*self.distance_to_g_], [K_att*theta_to_g]])
        
        objective = cp.Minimize((1 / 2) * cp.quad_form(U, H) - (H @ obj_function).T @ U)
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS)

        if U.value is not None and len(U.value) == 2:
            u, w = np.copy(U.value)
        else:
            u, w = [0.0, 0.0]
            print(f'QP solution not found at iteration {self.i_}')


        
        if abs(w) > 1:
            w = np.sign(w) * 1

        if abs(u) > 1:
            u = np.sign(u) * 1

        
       
        self.u_history_[:, 1:] = self.u_history_[:, :-1]  # Shift columns right
        self.u_history_[:, 0] = [u, w]                  # Insert new vector at column 0
        [u, w] = np.mean(self.u_history_, axis=1)       # Compute mean across columns
        
        [u, w] = [0.1*u, 0.1*w]
        return u, w
    
    def detect_dense_obstacle_cluster(self, laser_scan_data:LaserScan):   
        n= len(laser_scan_data.ranges)
        j = 0
        s = 0
        no_vide = 0
        for i in range(int(n/2),n-1):
            min_betw_points = 5*min(laser_scan_data.ranges[i],laser_scan_data.ranges[i + 1])*np.tan(laser_scan_data.angle_increment)
            if abs(laser_scan_data.ranges[i] - laser_scan_data.ranges[i + 1]) < min_betw_points:
                j += 1
            else:
                s += 1
                if s > int(0.4/min_betw_points):
                    break
        for i in range(int(n/2),0,-1):
            min_betw_points = 5*min(laser_scan_data.ranges[i],laser_scan_data.ranges[i - 1])*np.tan(laser_scan_data.angle_increment)
            if abs(laser_scan_data.ranges[i] - laser_scan_data.ranges[i - 1]) < min_betw_points:
                j += 1
            else:
                s += 1
                if s > int(0.4/min_betw_points):
                    break    
        no_vide = j >= 0.75*n
        return no_vide
 
    
    def create_vertual_obstacl(self, laser_scan_data:LaserScan):
        angl_list = np.arange(laser_scan_data.angle_min, laser_scan_data.angle_max + laser_scan_data.angle_increment, laser_scan_data.angle_increment)

        robot_to_p1_dist = laser_scan_data.ranges[int(len(laser_scan_data.ranges)/2)]
        robot_to_p1_angl = angl_list[int(len(laser_scan_data.ranges)/2)]
        p1 = [self.xt_[0,self.i_]+robot_to_p1_dist*np.cos(self.xt_[2,self.i_]+robot_to_p1_angl), self.xt_[1,self.i_]+robot_to_p1_dist*np.sin(self.xt_[2,self.i_]+robot_to_p1_angl)]
        
        robot_to_p2_dist = min(laser_scan_data.ranges[0:int(len(laser_scan_data.ranges)/2)])
        robot_to_p2_angl = angl_list[laser_scan_data.ranges.index(robot_to_p2_dist)]
        p2 = [self.xt_[0,self.i_]+robot_to_p2_dist*np.cos(self.xt_[2,self.i_]+robot_to_p2_angl), self.xt_[1,self.i_]+robot_to_p2_dist*np.sin(self.xt_[2,self.i_]+robot_to_p2_angl)]
        
        robot_to_p3_dist = min(laser_scan_data.ranges[int(len(laser_scan_data.ranges)/2):len(laser_scan_data.ranges)])
        robot_to_p3_angl = angl_list[laser_scan_data.ranges.index(robot_to_p3_dist)]
        p3 = [self.xt_[0,self.i_]+robot_to_p3_dist*np.cos(self.xt_[2,self.i_]+robot_to_p3_angl), self.xt_[1,self.i_]+robot_to_p3_dist*np.sin(self.xt_[2,self.i_]+robot_to_p3_angl)]

        origin = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3]  
        r = max(np.hypot(origin[0]-p2[0], origin[1]-p2[1]), np.hypot(origin[0]-p3[0], origin[1]-p3[1]), np.hypot(origin[0]-p1[0], origin[1]-p1[1]))
        vert_obs = Circle(origin, r)
        return  vert_obs

        
    
        
    def ploting_path(self):
        fig1, ax1 = plt.subplots(1)
        fig2, ax2 = plt.subplots(2) 
        img = mpimg.imread('/home/ihcene/Pictures/Screenshots/test3.png')
        ax1.imshow(img, extent=[-0.8, 5.2, -2.6, 2.4])      
        ax1.plot(self.xt_[0, :], self.xt_[1, :], linewidth=1.5, color='r')
        ax1.set_title('path')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.add_patch(plt.Circle([self.vert_obs_.origin[0], self.vert_obs_.origin[1]], self.vert_obs_.radius, color='r'))
        ax1.add_patch(plt.Circle([Xgoal[0], Xgoal[1]], 0.05, color='y'))
        ax2[0].plot(self.ut_[0, :], linewidth=1, color='g')
        ax2[0].set_title('linear velocity')
        ax2[0].set_xlabel('iterations')
        ax2[0].set_ylabel('m/s')
        ax2[1].plot(self.ut_[1, :], linewidth=1, color='g')
        ax2[1].set_title('angular velocity')
        ax2[1].set_xlabel('iterations')
        ax2[1].set_ylabel('rad/s')
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    cbf_node = CBF()
    rclpy.spin(cbf_node)
    cbf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()