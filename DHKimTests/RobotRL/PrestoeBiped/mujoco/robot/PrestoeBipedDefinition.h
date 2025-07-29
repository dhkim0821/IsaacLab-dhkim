#ifndef __PRESTOE_BIPED_DEFINTION_H__
#define __PRESTOE_BIPED_DEFINTION_H__

#include <iostream>
#include <vector>

namespace prestoebiped
{
    constexpr size_t num_act_joints = 15;
    constexpr size_t num_leg_joints = 7;
    constexpr size_t num_legs = 2;
    constexpr size_t num_arm_joint = 0;
    constexpr size_t nDoF = num_act_joints + 6;
    constexpr size_t num_contacts = 10;
    constexpr size_t num_jencoder_boards = 2;
    constexpr size_t num_jencs_per_board = 2;

    constexpr size_t RKneeIdx = 4;
    constexpr size_t RAnklePitchIdx = 5;
    constexpr size_t LKneeIdx = 11;
    constexpr size_t LAnklePitchIdx = 12;

    constexpr float toe_width = 0.06;
    constexpr float toe_length = 0.09;

    //mapping from how acuators actually connected vs FB model
    static std::vector<int> right_actuator_indices_map = {2, 0, 3, 4, 1, 5, 6};
    static std::vector<int> left_actuator_indices_map = {12, 10, 13, 8, 11, 9, 14};
    static std::vector<int> full_actuator_indices_map = {7, 2, 0, 3, 4, 1, 5, 6, 12, 10, 13, 8, 11, 9, 14};
    // static float JOINT_OFFSETS[15] = {0.0f, -0.261799f, 0.0f, -1.4407693f, -11.63551111f, -15.3404f, -15.3430f, -14.5485f, 0.261799f, 0.0f, 1.4407693f, -11.63551111f, 15.3404f, 15.3404f, 14.5485f}; // TUNED
    // static float JOINT_OFFSETS[15] = {0.0f, -0.261799f, 0.0f, -1.4407693f, -11.63551111f, -15.3404f, -15.3430f, -14.5485f, 0.261799f, 0.0f, 1.4407693f, -11.63551111f, -15.3404f, -15.3404f, -14.5485f}; // WORKING
    static float JOINT_OFFSETS[15] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

} // namespace prestoe_biped

/*!
 * Representation of a Prestoe robot's physical properties.
 * Leg
 * 1 0   RIGHT
 * 3 2   RIGHT
 * Arm
 *
 */

namespace prestoebiped_fb_link { // numbering depends on order of how contact points are added
  constexpr size_t rfoot = 12; 
  constexpr size_t rtoe = 13; 
  constexpr size_t lfoot = 19; 
  constexpr size_t ltoe = 20; 

  // constexpr size_t rlowarm = 24;
  // constexpr size_t llowarm = 28;
}

//  Foot index
//  1 --- 3
//  |     |
//  2 --- 4
//   \   /
//    | |
//    heel
namespace prestoebiped_contact{
  constexpr size_t rheel = 8; 
  constexpr size_t rtoe_1 = 9;
  constexpr size_t rtoe_2 = 10;
  constexpr size_t rtoe_3 = 11;
  constexpr size_t rtoe_4 = 12;

  constexpr size_t lheel = 13; 
  constexpr size_t ltoe_1 = 14;
  constexpr size_t ltoe_2 = 15;
  constexpr size_t ltoe_3 = 16;
  constexpr size_t ltoe_4 = 17;

  constexpr size_t rtoe = 18;
  constexpr size_t ltoe = 19;

  constexpr size_t num_foot_contacts = 10;

  // constexpr size_t rhand = 18;
  // constexpr size_t lhand = 19;
}

#endif