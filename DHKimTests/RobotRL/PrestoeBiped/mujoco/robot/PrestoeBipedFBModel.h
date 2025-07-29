#ifndef __PRESTOE_BIPED_FLOATING_BASE_MODEL_H__
#define __PRESTOE_BIPED_FLOATING_BASE_MODEL_H__ 

#include <FloatingBaseModel.h>
#include <parsingURDF.h>
#include <Configuration.h>
#include <PrestoeBipedDefinition.h>

template <typename T>
class PrestoeBipedFBModel {
  public:
    static void buildFBModel(FloatingBaseModel<T> & model, bool verbose = false, T gravity = -9.81)
    {
      buildFloatingBaseModelFromURDF(model, 
      THIS_COM"/Systems/PrestoeBipedSystem/Robot/prestoe_biped.urdf", verbose);

      // Contact setup
      Vec3<T> offset;
      offset.setZero();
      Vec3<T> waistDims(0.2, 0.2, 0.3);
      offset[2] = waistDims[2]/2.0; 
      model.addGroundContactBoxPointsOffset(5, waistDims, offset);

      // right foot contact
      model.addGroundContactPoint(prestoebiped_fb_link::rfoot, Vec3<T>(-0.055, 0.0, -0.04));
      model.addGroundContactPoint(prestoebiped_fb_link::rtoe, Vec3<T>(0.075, 0.03, -0.031));
      model.addGroundContactPoint(prestoebiped_fb_link::rtoe, Vec3<T>(-0.02, 0.03, -0.031));
      model.addGroundContactPoint(prestoebiped_fb_link::rtoe, Vec3<T>(0.075, -0.03, -0.031));
      model.addGroundContactPoint(prestoebiped_fb_link::rtoe, Vec3<T>(-0.02, -0.03, -0.031));

      // left foot contact
      model.addGroundContactPoint(prestoebiped_fb_link::lfoot, Vec3<T>(-0.055, 0.0, -0.04));
      model.addGroundContactPoint(prestoebiped_fb_link::ltoe, Vec3<T>(0.075, 0.03, -0.031));
      model.addGroundContactPoint(prestoebiped_fb_link::ltoe, Vec3<T>(-0.02, 0.03, -0.031));
      model.addGroundContactPoint(prestoebiped_fb_link::ltoe, Vec3<T>(0.075, -0.03, -0.031));
      model.addGroundContactPoint(prestoebiped_fb_link::ltoe, Vec3<T>(-0.02, -0.03, -0.031));

      // Toe (center)
      model.addGroundContactPoint(prestoebiped_fb_link::rtoe, Vec3<T>(0.027, 0.0, -0.031));
      model.addGroundContactPoint(prestoebiped_fb_link::ltoe, Vec3<T>(0.027, 0.0, -0.031));

      Vec3<T> g(0, 0, gravity);
      model.setGravity(g);
    }
};

#endif // LIBBIOMIMETICS_Prestoe_H
