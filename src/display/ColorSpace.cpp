#include "display/ColorSpace.h"

namespace display {

const Mat3 kRec709ToRec2020 = {{
    {0.6274040f, 0.3292820f, 0.0433136f},
    {0.0690970f, 0.9195400f, 0.0113612f},
    {0.0163916f, 0.0880132f, 0.8955950f},
}};

const Mat3 kRec2020ToRec709 = {{
    {1.6604910f, -0.5876411f, -0.0728499f},
    {-0.1245505f, 1.1328999f, -0.0083494f},
    {-0.0181508f, -0.1005789f, 1.1187297f},
}};

const Mat3 kIdentity3 = {{
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
}};

const Mat3& workingToRec2020(WorkingColorSpace space)
{
    return space == WorkingColorSpace::Rec709 ? kRec709ToRec2020 : kIdentity3;
}

glm::vec3 luminanceWeights(WorkingColorSpace space)
{
    return space == WorkingColorSpace::Rec709 ? glm::vec3(0.2126f, 0.7152f, 0.0722f)
                                              : glm::vec3(0.2627f, 0.6780f, 0.0593f);
}

const char* toString(WorkingColorSpace space)
{
    return space == WorkingColorSpace::Rec709 ? "Rec709" : "Rec2020";
}

glm::vec3 mul(const Mat3& a, const glm::vec3& v)
{
    return {a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
            a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
            a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z};
}

} // namespace display
