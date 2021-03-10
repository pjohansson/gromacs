/*! \brief Direction along which to define something
 */
enum eFlowAxis
{
    eFlowAxisX,
    eFlowAxisY,
    eFlowAxisZ,
    eFlowAxisTypesNR
};
//! Names for swapping
extern const char* eFlowAxisTypes_names[eFlowAxisTypesNR + 1];
//! Macro for swapping string
#define EFLOWAXISTYPE(e) enum_name(e, eFlowAxisTypesNR, eFlowAxisTypes_names)

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
//! Macro for swapping string
#define EFLOWSWAPPOSITIONAXISTYPE(e) enum_name(e, eFlowSwapPositionAxisTypesNR, eFlowSwapPositionAxisTypes_names)

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