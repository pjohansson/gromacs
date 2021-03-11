#include "gromacs/math/vectypes.h"
#include "enums.h"

const char* eFlowSwapMethodTypes_names[static_cast<size_t>(eFlowSwapMethod::NR) + 1] = {
    "center-edge",
    "positions-relative",
    "positions-absolute",
    nullptr
};
const char* eFlowSwapAxisTypes_names[eFlowSwapAxisTypesNR + 1] = { "X", "Y", "Z", nullptr };
const char* eFlowSwapPositionAxisTypes_names[eFlowSwapPositionAxisTypesNR + 1] = { "Z", "X", "Y", nullptr };

const char* eFlowSwapPosition2String(const int axis) 
{
    size_t correct_axis = XX;

    switch (axis)
    {
        case XX:
            correct_axis = eFlowSwapPositionAxisX;
            break;

        case YY:
            correct_axis = eFlowSwapPositionAxisY;
            break;

        case ZZ:
            correct_axis = eFlowSwapPositionAxisZ;
            break;
    }

    return eFlowSwapPositionAxisTypes_names[correct_axis];
}