#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3

v = 0.1  # Velocidade linear
w = 1  # Velocidade angular

def praFrente(x):
	speed = Twist(Vector3(x,0,0), Vector3(0,0,0))
	pub.publish(speed)

def praTras(x):
	speed = Twist(Vector3(-x,0,0), Vector3(0,0,0))
	pub.publish(speed)	
	
def proLado(x):
	speed = Twist(Vector3(0,0,0),Vector3(0,0,x))
	pub.publish(speed)

def proOutroLado(x):
	speed = Twist(Vector3(0,0,0),Vector3(0,0,-x))
	pub.publish(speed)

def para():
	speed = Twist(Vector3(0,0,0), Vector3(0,0,0))
	pub.publish(speed)

if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)

    try:
        while not rospy.is_shutdown():
			rospy.sleep(5)
			praFrente(v)
			rospy.sleep(10)
			para()
			rospy.sleep(5)
			proLado(w)
			rospy.sleep(1.6)
			para()

    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")
