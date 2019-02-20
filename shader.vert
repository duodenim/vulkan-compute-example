#version 450

out gl_PerVertex
{
  vec4 gl_Position;
  float gl_PointSize;
};

const vec2 positions[3] = {
  {-0.5, 0.5},
  {0.5, 0.5},
  {0.0, -0.5}
};

#define NUM_PARTICLES 100
layout (std430, binding = 0) buffer buf {
  readonly vec4 values[400];
  readonly vec4 simulation_params;
};

void main() {
  gl_Position = vec4(values[gl_VertexIndex].xyz, 1.0);
  gl_PointSize = 4.0;
}
