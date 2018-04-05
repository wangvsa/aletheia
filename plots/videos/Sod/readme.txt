0. sod_clean.mp4
    a clean run of Sod program, 200 iterations

1. sod_error_50.mp4:
    Inject error at 50th iterations then run 100 iterations
    Error position: window=41, x=145, y=342
    old:, 0.42674601358689623
    new:, 9.9359549020160044e-11

2. sod_error_70.mp4:
    Inject error at 70th iterations then run 100 iterations
    Error position: window=12, x=65, y=80
    old: 1.0
    new: 0.0625

3. sod_error_80.mp4:
    Inject error at 80th iterations then run 100 iterations
    Error position: window=102, x=390, y=147
    old: 0.14473335902070955
    new: 0.00056536468367464668

4. sod_error_90.mp4:
    Inject error at 90th iterations then run 100 iterations
    Error position: window=100, x=397, y=75
    old: 0.47148243180679977
    new: 0.48710743180679977

5. sod_error_90_diff.mp4:
    draw the difference between clean run and error_90

6. sod_error_visit.mp4:
    inject error at 50th iteration then run 100 iterations
    note: the error is not the same as int sod_error_50>.mp4
    like sod_error_90, the error is not visible if drawn with matplotlib
    but can see it if visualize it with visit
