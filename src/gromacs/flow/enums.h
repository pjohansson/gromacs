#ifndef FLOW_ENUMS_H
#define FLOW_ENUMS_H

#include <cstddef>

//! \brief Method for specifying zone positions along the swap axis
enum class eFlowSwapMethod 
{
    CenterEdge,
    PositionsRelative,
    PositionsAbsolute,
    NR,
};
extern const char* eFlowSwapMethodTypes_names[static_cast<size_t>(eFlowSwapMethod::NR) + 1];
#define EFLOWSWAPMETHODTYPE(e) enum_name(static_cast<int>(e), static_cast<int>(eFlowSwapMethod::NR), eFlowSwapMethodTypes_names)

/*! \brief Direction along which to construct swap zones
 */
enum eFlowSwapPositionAxis
{
    eFlowSwapPositionAxisZ,
    eFlowSwapPositionAxisX,
    eFlowSwapPositionAxisY,
    eFlowSwapPositionAxisTypesNR
};
//! Names for swapping
extern const char* eFlowSwapPositionAxisTypes_names[eFlowSwapPositionAxisTypesNR + 1];
const char* eFlowSwapPosition2String(const int axis);

/*! \brief Direction along which to swap molecules
 */
enum eFlowSwapAxis
{
    eFlowSwapAxisX,
    eFlowSwapAxisY,
    eFlowSwapAxisZ,
    eFlowSwapAxisTypesNR
};
//! Names for swapping
extern const char* eFlowSwapAxisTypes_names[eFlowSwapAxisTypesNR + 1];
//! Macro for swapping string
#define EFLOWSWAPAXISTYPE(e) enum_name(e, eFlowSwapAxisTypesNR, eFlowSwapAxisTypes_names)

#endif // FLOW_ENUMS_H