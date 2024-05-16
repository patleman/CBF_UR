import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import numpy as np
import tf_transformations
import tf2_ros
import asyncio
from velocity_profile.qp_control import CBFQPSolver
from sensor_msgs.msg import JointState

import csv




class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self,spin_thread=True)
        self.solver = CBFQPSolver(6, 0, 3)
        self.pe_values = []
        self.joint_pa =  []
        self.joint_va =  []
        self.joint_pd =  []
        self.joint_vd  = []
        self.error_prev=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        self.error_i=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        
        #joint_space
        self.thetas0 = np.array([-2.02, -2.30, -1.52, -0.82, 1.64, 0])  # point 1
        self.thetas1 = np.array([-2.02, -2.01, -1.24, -0.82, 1.64, 0])  # point 2
        self.thetas2 = np.array([-1.19, -1.71, -1.24, -1.61, 1.48, 0])  # point 3
        self.thetas3 = np.array([-0.59, -1.80, -1.25, -1.59, 1.62, 0])  # point 4
        self.thetas4 = np.array([-0.59, -2.26, -1.82, -0.65, 1.433, 0]) # point 5
        
        #operational_space
        # self.thetas0 = np.array([0.458, 0.589, 0.215])  # point 1
        # self.thetas1 = np.array([0.457, 0.587, 0.552])  # point 2
        # self.thetas2 = np.array([-0.035, 0.586, 0.620])  # point 3
        # self.thetas3 = np.array([-0.399, 0.469, 0.572])  # point 4
        # self.thetas4 = np.array([-0.424, 0.512, 0.138]) # point 5

        self.waypoints = np.array([self.thetas0, self.thetas1, self.thetas2, self.thetas3, self.thetas4]).T
        self.t0=0
        self.t1=4
        self.t2=8
        self.t3=12
        self.t4=16

        self.dt1=0
        self.dt2=0.2
        self.dt3=0.4
        self.dt4=0.6

        # Defining tangent unit vectors
        self.T1 = (self.thetas1 - self.thetas0) / np.linalg.norm(self.thetas1 - self.thetas0)
        self.T2 = (self.thetas2 - self.thetas1) / np.linalg.norm(self.thetas2 - self.thetas1)
        self.T3 = (self.thetas3 - self.thetas2) / np.linalg.norm(self.thetas3 - self.thetas2)
        self.T4 = (self.thetas4 - self.thetas3) / np.linalg.norm(self.thetas4 - self.thetas3)

        # Structuring Data
        self.T = np.column_stack((self.T1, self.T2, self.T3, self.T4))  # list of tangent unit vectors

        # list of time instances with time advancements
        self.ta = np.array([self.t0, self.t1 - self.dt1, self.t2 - self.dt2, self.t3 - self.dt3, self.t4 - self.dt4])

        # list of time instances without time advancements
        self.ti = np.array([self.t0, self.t1 - self.dt2, self.t2 - self.dt3, self.t3 - self.dt4])

        self.Na = np.array([np.linalg.norm(self.thetas1 - self.thetas0),  # list of norms of segments
                    np.linalg.norm(self.thetas2 - self.thetas1),
                    np.linalg.norm(self.thetas3 - self.thetas2),
                    np.linalg.norm(self.thetas4 - self.thetas3)])
        
        self.source_frame = 'base_link'
        self.target_frames = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link']

        # self.q_init=np.array([-2.02,-2.30,-1.52,-0.82,1.64,0])
        # self.qinit_state=0
        # self.prevtime=0

        self.start_time_ = self.get_clock().now().to_msg()
        #self.timer_ = self.create_timer(0.01, self.publish_joint_positions)
        self.sub_=self.create_subscription(JointState, "joint_states",self.JointstateCallback,10)
        self.get_logger().info('Publishing joint velocities')

    def JointstateCallback(self,msg):
        # print(msg.position)
        joint_vs=np.array([msg.velocity[5],msg.velocity[0],msg.velocity[1],msg.velocity[2],msg.velocity[3],msg.velocity[4]])
        joint_s=np.array([msg.position[5],msg.position[0],msg.position[1],msg.position[2],msg.position[3],msg.position[4]])
        self.joint_va.append(joint_vs)
        self.joint_pa.append(joint_s)
        self.publish_joint_positions(msg.position)
       
        




    def publish_joint_positions(self,joint_position):
        current_time = self.get_clock().now().to_msg()
        elapsed_time = (current_time.sec - self.start_time_.sec) + (current_time.nanosec - self.start_time_.nanosec) * 1e-9
        joint_s=np.array([joint_position[5],joint_position[0],joint_position[1],joint_position[2],joint_position[3],joint_position[4]])
        message = Float64MultiArray()
        
        #trajectory_generation
        Pd,Pd_dot=self.compute_pd_and_pd_dot(elapsed_time)

        self.joint_pd.append(Pd)
        self.joint_vd.append(Pd_dot)
        
        # Calculate position based on trapezoidal profile
        
        ## safety filter comes here 

        # get jacobian

        # Compute Jacobian
        

        # print(Jacobian)
        p_obs=np.array([-0.034,0.585,0.627])
        pe=self.get_end_effector_position()
        self.pe_values.append(pe)
        distance = np.linalg.norm(p_obs-pe)
        if(distance<1.35):
            Jacobian,pe = self.compute_jacobian()
            Jacobian_1,pe1=self.compute_jacobian_1()
            Jacobian_2,pe2=self.compute_jacobian_2()

            curr_error=np.array([(Pd[0]-joint_s[0]), (Pd[1]-joint_s[1]), (Pd[2]-joint_s[2]), (Pd[3]-joint_s[3]), (Pd[4]-joint_s[4]), (Pd[5]-joint_s[5])])
            e_d=curr_error-self.error_prev
            self.error_i=+curr_error
            u_normal= np.array([Pd_dot[0]+1.2*(Pd[0]-joint_s[0])+0.6*(e_d[0])+0.02*(self.error_i[0]), Pd_dot[1]+1.2*(Pd[1]-joint_s[1])+0.6*(e_d[1])+0.02*(self.error_i[1]), Pd_dot[2]+1.2*(Pd[2]-joint_s[2])+0.6*(e_d[2])+0.02*(self.error_i[2]), Pd_dot[3]+1.2*(Pd[3]-joint_s[3])+0.6*(e_d[3])+0.02*(self.error_i[3]), Pd_dot[4]+1.2*(Pd[4]-joint_s[4])+0.6*(e_d[4])+0.02*(self.error_i[4]), Pd_dot[5]+1.2*(Pd[5]-joint_s[5])+0.6*(e_d[5])+0.02*(self.error_i[5])])
            self.error_prev=curr_error
            # u_normal= np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

            p_err=pe-p_obs
            p_err1=pe1-p_obs
            p_err2=pe2-p_obs
            # now solve QP
            q_safe=self.QP_solver(u_normal,Jacobian,Jacobian_1,Jacobian_2,p_err,p_err1,p_err2)
            message.data = [q_safe[0], q_safe[1], q_safe[2], q_safe[3], q_safe[4], q_safe[5]]
            self.publisher_.publish(message)
        
        else:
            curr_error=np.array([(Pd[0]-joint_s[0]), (Pd[1]-joint_s[1]), (Pd[2]-joint_s[2]), (Pd[3]-joint_s[3]), (Pd[4]-joint_s[4]), (Pd[5]-joint_s[5])])
            e_d=curr_error-self.error_prev
            self.error_i=+curr_error
            message.data= [Pd_dot[0]+3.2*(Pd[0]-joint_s[0])+0.6*(e_d[0])+0.02*(self.error_i[0]), Pd_dot[1]+3.2*(Pd[1]-joint_s[1])+0.6*(e_d[1])+0.02*(self.error_i[1]), Pd_dot[2]+3.2*(Pd[2]-joint_s[2])+0.6*(e_d[2])+0.02*(self.error_i[2]), Pd_dot[3]+3.2*(Pd[3]-joint_s[3])+0.6*(e_d[3])+0.02*(self.error_i[3]), Pd_dot[4]+3.2*(Pd[4]-joint_s[4])+0.6*(e_d[4])+0.02*(self.error_i[4]), Pd_dot[5]+3.2*(Pd[5]-joint_s[5])+0.6*(e_d[5])+0.02*(self.error_i[5])]
            self.error_prev=curr_error
            # = [Pd_dot[0]+3.2*(Pd[0]-joint_s[0]), Pd_dot[1]+3.2*(Pd[1]-joint_s[1]), Pd_dot[2]+3.2*(Pd[2]-joint_s[2]), Pd_dot[3]+3.2*(Pd[3]-joint_s[3]), Pd_dot[4]+3.2*(Pd[4]-joint_s[4]), Pd_dot[5]+3.2*(Pd[5]-joint_s[5])]
         
           
            self.publisher_.publish(message)

        # if elapsed_time >= 22:
        #     self.timer_.cancel()
    def QP_solver(self,u_des,jacobian,jacobian_1,jacobian_2,p_error,p_error1,p_error2):
        params = {
            "Jacobian": jacobian,
            "Jacobian1":jacobian_1,
            "Jacobian2":jacobian_2,
            "p_error": p_error,
            "p_error1": p_error1,
            "p_error2":p_error2,
            "u_current": u_des,
        }

        # solver for target joint velocity
        self.solver.solve(params)
        dq_target = self.solver.qp.results.x
        return dq_target

        
    def get_transform_matrix(self, source_frame, target_frame):
        try:
            # transform_stamped = self.tf_buffer.wait_for_transform_async(
            #     source_frame, target_frame, rclpy.time.Time()
            # )
            # rclpy.spin_until_future_complete(self, transform_stamped)
            # print("Got transform from {} to {}!".format(source_frame, target_frame))
            # print("just_before")
            transform_stamped = asyncio.run(
                self.tf_buffer.lookup_transform_async(
                    source_frame, target_frame, rclpy.time.Time(seconds=0)
                )
            )
            transform_matrix = self.transform_to_matrix(transform_stamped.transform)
            return transform_matrix
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().error('Failed to lookup transform: %s' % e)
            return None  
          
    def transform_to_matrix(self, transform):
        translation = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z]
        )
        rotation = np.array(
            [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        )
        rotation_matrix = tf_transformations.quaternion_matrix(rotation)
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation
        transform_matrix = np.dot(translation_matrix, rotation_matrix)
        return transform_matrix
    def get_end_effector_position(self):
        source_frame = 'base_link' # Replace with your source frame ID
        target_frame = 'tool0' # Replace with your target frame ID
        e_transform_matrix = self.get_transform_matrix(source_frame, target_frame)
        if e_transform_matrix is not None:
            # print("Transformation matrix from {} to {}: \n{}".format(source_frame, target_frame, e_transform_matrix))
            p6=e_transform_matrix[:3,3]
            return p6  
        else:
            return None  
        
    def compute_jacobian_1(self):
        # Get transformation matrices for each target frame
        source_frame = 'base_link' # Replace with your source frame ID
        target_frame = 'wrist_2_link' # Replace with your target frame ID
        z0=np.array([0,0,1]).T
        p0=np.array([0,0,0]).T

        e_transform_matrix = self.get_transform_matrix(source_frame, target_frame)
        if e_transform_matrix is not None:
            # print("Transformation matrix from {} to {}: \n{}".format(source_frame, target_frame, e_transform_matrix))
            p6=e_transform_matrix[:3,3]
        # print("reached 2")

        transform_matrices = []
        
        target_frames = ['shoulder_link', 'upper_arm_link','forearm_link', 'wrist_1_link' ]
        for target_frame in target_frames:
            transform_matrix = self.get_transform_matrix(source_frame, target_frame)
            if transform_matrix is not None:
                transform_matrices.append(transform_matrix)
        
        # Compute Jacobian
        
        Jacobian = [np.cross(z0, (p6 - p0))]
        for transform_matrix in transform_matrices:
            p = transform_matrix[:3, 3]
            z = transform_matrix[:3, 2]
            Jacobian.append(np.cross(z, (p6 - p)))
        # Jacobian.append([0,0,0])
        # Jacobian.append([0,0,0])
        Jacobian.append([0,0,0])
        Jacobian = np.array(Jacobian).T
        
        return Jacobian,p6
    def compute_jacobian_2(self):
        # Get transformation matrices for each target frame
        source_frame = 'base_link' # Replace with your source frame ID
        target_frame = 'wrist_1_link' # Replace with your target frame ID
        z0=np.array([0,0,1]).T
        p0=np.array([0,0,0]).T

        e_transform_matrix = self.get_transform_matrix(source_frame, target_frame)
        if e_transform_matrix is not None:
            # print("Transformation matrix from {} to {}: \n{}".format(source_frame, target_frame, e_transform_matrix))
            p6=e_transform_matrix[:3,3]
        # print("reached 2")
        # source_frame = 'base_link' # Replace with your source frame ID
        # target_frame = 'forearm_link' # Replace with your target frame ID
        # e_transform_matrix = self.get_transform_matrix(source_frame, target_frame)
        # if e_transform_matrix is not None:
        #     # print("Transformation matrix from {} to {}: \n{}".format(source_frame, target_frame, e_transform_matrix))
        #     p7=e_transform_matrix[:3,3]

        # p6=(p6+p7)/2  
        transform_matrices = []
        
        target_frames = ['shoulder_link', 'upper_arm_link','forearm_link' ]
        for target_frame in target_frames:
            transform_matrix = self.get_transform_matrix(source_frame, target_frame)
            if transform_matrix is not None:
                transform_matrices.append(transform_matrix)
        
        # Compute Jacobian
        
        Jacobian = [np.cross(z0, (p6 - p0))]
        for transform_matrix in transform_matrices:
            p = transform_matrix[:3, 3]
            z = transform_matrix[:3, 2]
            Jacobian.append(np.cross(z, (p6 - p)))
        # Jacobian.append([0,0,0])
        Jacobian.append([0,0,0])
        Jacobian.append([0,0,0])
        Jacobian = np.array(Jacobian).T
        
        return Jacobian,p6
    def compute_jacobian(self):
        # Get transformation matrices for each target frame
        source_frame = 'base_link' # Replace with your source frame ID
        target_frame = 'tool0' # Replace with your target frame ID
        z0=np.array([0,0,1]).T
        p0=np.array([0,0,0]).T

        e_transform_matrix = self.get_transform_matrix(source_frame, target_frame)
        if e_transform_matrix is not None:
            # print("Transformation matrix from {} to {}: \n{}".format(source_frame, target_frame, e_transform_matrix))
            p6=e_transform_matrix[:3,3]
        # print("reached 2")

        transform_matrices = []
        for target_frame in self.target_frames:
            transform_matrix = self.get_transform_matrix(source_frame, target_frame)
            if transform_matrix is not None:
                transform_matrices.append(transform_matrix)
        
        # Compute Jacobian
      
        Jacobian = [np.cross(z0, (p6 - p0))]
        for transform_matrix in transform_matrices:
            p = transform_matrix[:3, 3]
            z = transform_matrix[:3, 2]
            Jacobian.append(np.cross(z, (p6 - p)))
        Jacobian = np.array(Jacobian).T
        return Jacobian,p6  
      
    def Trapezoidal_p(self,t, qf, qi, tf, ti):
        t = t - ti  # taking difference to use time in formula
        tf = tf - ti
        qc_dot_dot = math.pi / 6  # greater than 4*(qf-qi)/tf**2

        tc = ((tf) / 2) - (1 / 2) * math.sqrt((((tf ** 2) * qc_dot_dot) - 4 * (qf - qi)) / qc_dot_dot)

        if (t <= tc and t >= 0):
            pt = qi + (0.5) * qc_dot_dot * t ** 2
        elif (t > tc and t <= tf - tc):
            pt = qi + qc_dot_dot * tc * (t - (tc / 2))
        elif (t >= tf - tc and t <= (tf)):
            pt = qf - (0.5) * qc_dot_dot * (((tf) - t) ** 2)
        else:
            pt=0    
        
        return pt

    def Trapezoidal_pd(self,t, qf, qi, tf, ti):
        t = t - ti  # taking difference to use time in formula
        tf = tf - ti
        qc_dot_dot = math.pi / 6 # greater than 4*(qf-qi)/tf**2

        tc = ((tf) / 2) - (0.5) * math.sqrt(((((tf) ** 2) * qc_dot_dot) - 4 * (qf - qi)) / qc_dot_dot)

        if (t <= tc and t >= 0):
            ptd = qc_dot_dot * t
        elif (t > tc and t <= tf - tc):
            ptd = qc_dot_dot * tc
        elif (t >= tf - tc and t <= tf):
            ptd = qc_dot_dot * (tf - t)
        else:
            ptd=0    
        
        return ptd
    def compute_pd_and_pd_dot(self,t):

        if 0 <= t < self.t1 - self.dt2:  # only s1
            j = 0
            pd = self.waypoints[:, j] + self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot = self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]

        elif self.t1 - self.dt2 <= t < self.t1 - self.dt1:  # s1 and s2
            j = 0
            pd = self.waypoints[:, j] + self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot = self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]

            j = 1
            pd += self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot += self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


        elif self.t1 - self.dt1 <= t < self.t2 - self.dt3:  # only s2
            j = 1
            pd = self.waypoints[:, j] + self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot = self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


        elif self.t2 - self.dt3 <= t < self.t2 - self.dt2:  # s2 and s3
            j = 1
            pd = self.waypoints[:, j] + self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot = self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


            j = 2
            pd +=  self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot += self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


        elif self.t2 - self.dt2 <= t < self.t3 - self.dt4:  # only s3
            j = 2
            pd = self.waypoints[:, j] + self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot = self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


        elif self.t3 - self.dt4 <= t < self.t3 - self.dt3:  # s3 and s4
            j = 2
            pd = self.waypoints[:, j] + self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot = self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


            j = 3
            pd +=  self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot += self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


        else:  # only s4
            j = 3
            pd = self.waypoints[:, j] + self.Trapezoidal_p(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]
            pd_dot = self.Trapezoidal_pd(t, self.Na[j], 0, self.ta[j + 1], self.ti[j]) * self.T[:, j]


        return pd, pd_dot
        

def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    # rclpy.spin(node)
    # rclpy.shutdown()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        
        # Save values to a CSV file
        with open('pe_values.csv', 'w', newline='') as csvfile: #end-effector position saved
            csvwriter = csv.writer(csvfile)
            for pe in node.pe_values:
                csvwriter.writerow(pe)

        with open('joint_pa_values.csv', 'w', newline='') as csvfile: # each actual joint angle saved
            csvwriter = csv.writer(csvfile)
            for pa in node.joint_pa:
                csvwriter.writerow(pa)   

        with open('joint_pd_values.csv', 'w', newline='') as csvfile:  # each desired joint angle saved
            csvwriter = csv.writer(csvfile)
            for pd in node.joint_pd:
                csvwriter.writerow(pd)   
        with open('joint_va_values.csv', 'w', newline='') as csvfile:   # each desired joint velocity saved
            csvwriter = csv.writer(csvfile)
            for va in node.joint_va:
                csvwriter.writerow(va) 
        with open('joint_vd_values.csv', 'w', newline='') as csvfile:# each actual joint velocity saved 
            csvwriter = csv.writer(csvfile)
            for vd in node.joint_vd:
                csvwriter.writerow(vd)                            

        # Plot the pe values
        print("CSV file written")    
        pass
    finally:
        rclpy.shutdown()

        

if __name__ == '__main__':
    main()