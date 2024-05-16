import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import time
import asyncio
import tf_transformations


class TransformMatrixNode(Node):
    def __init__(self):
        super().__init__('transform_matrix_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self,spin_thread=True)

        self.timer_period = 0.1  # in seconds
        self.timer = self.create_timer(self.timer_period, self.joint_publisher)

    def joint_publisher(self):
        source_frame = 'base_link' # Replace with your source frame ID
        target_frame = 'tool0' # Replace with your target frame ID
        z0=np.array([0,0,1]).T
        p0=np.array([0,0,0]).T

        e_transform_matrix = self.get_transform_matrix(source_frame, target_frame)
        if e_transform_matrix is not None:
            print("Transformation matrix from {} to {}: \n{}".format(source_frame, target_frame, e_transform_matrix))
            p6=e_transform_matrix[:3,3]


    
        baseLink_T_shoulder_link=self.get_transform_matrix(source_frame, 'shoulder_link')
        if baseLink_T_shoulder_link is not None:
            print("Transformation matrix base_shoulder = \n{}".format(baseLink_T_shoulder_link))
            z1=baseLink_T_shoulder_link[:3,2]
            p1=baseLink_T_shoulder_link[:3,3]

        baseLink_T_upper_arm_link=self.get_transform_matrix(source_frame, 'upper_arm_link')
        if baseLink_T_upper_arm_link is not None:
            print("Transformation matrix  base_upper_arm= \n{}".format(baseLink_T_upper_arm_link))  
            z2=baseLink_T_upper_arm_link[:3,2]
            p2=baseLink_T_upper_arm_link[:3,3]


        baseLink_T_forearm_link=self.get_transform_matrix(source_frame, 'forearm_link')
        if baseLink_T_forearm_link is not None:
            print("Transformation matrix base_forearm = \n{}".format(baseLink_T_forearm_link))   
            z3=baseLink_T_forearm_link[:3,2]
            p3=baseLink_T_forearm_link[:3,3]  

        baseLink_T_wrist_1_link=self.get_transform_matrix(source_frame, 'wrist_1_link')
        if baseLink_T_wrist_1_link is not None:
            print("Transformation matrix base_wrist1 = \n{}".format(baseLink_T_wrist_1_link))  
            z4=baseLink_T_wrist_1_link[:3,2]
            p4=baseLink_T_wrist_1_link[:3,3]     

        baseLink_T_wrist_2_link=self.get_transform_matrix(source_frame, 'wrist_2_link')
        if baseLink_T_wrist_2_link is not None:
            print("Transformation matrix base_wrist2= \n{}".format(baseLink_T_wrist_2_link))  
            z5=baseLink_T_wrist_2_link[:3,2]
            p5=baseLink_T_wrist_2_link[:3,3] 
        
        baseLink_T_wrist_3_link=self.get_transform_matrix(source_frame, 'wrist_3_link')
        if baseLink_T_wrist_3_link is not None:
            print("Transformation matrix base_wrist3 = \n{}".format(baseLink_T_wrist_3_link))   


        Jacobian=np.array([np.cross(z0,(p6-p0)), np.cross(z1,(p6-p1)), np.cross(z2,(p6-p2)) ,np.cross(z3,(p6-p3)) ,np.cross(z4,(p6-p4)) ,np.cross(z5,(p6-p5)) ]).T 

        print(Jacobian)     

        print(z0,z1,z2,z3,z4,z5)      

    def get_transform_matrix(self, source_frame, target_frame):
       
        try:
            # transform_stamped = self.tf_buffer.wait_for_transform_async(source_frame,
            # target_frame, rclpy.time.Time())
            # rclpy.spin_until_future_complete(self, transform_stamped)
           # print("Got it!")
            transform_stamped = asyncio.run(self.tf_buffer.lookup_transform_async(
                    source_frame,target_frame,
                    rclpy.time.Time()
                ))
            # print(transform_stamped)
            transform_matrix = self.transform_to_matrix(transform_stamped.transform)
            return transform_matrix
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error('Failed to lookup transform: %s' % e)
            
    def transform_to_matrix(self, transform):
        translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
        rotation = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
        rotation_matrix = tf_transformations.quaternion_matrix(rotation)
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation
        transform_matrix = np.dot(translation_matrix, rotation_matrix)
        return transform_matrix

def main(args=None):
    rclpy.init(args=args)
    node = TransformMatrixNode()
    
          




    
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
