#include "optix/shared.h"

RT_PROGRAM
void anyhit() {
  rtTerminateRay();
}
