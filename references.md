# References

Sources for the methods, standards and software used in this project.

## Visual servoing
1. F. Chaumette and S. Hutchinson, "Visual servo control. I. Basic approaches," *IEEE Robotics & Automation Magazine*, vol. 13, no. 4, pp. 82–90, Dec. 2006. (Point-feature interaction matrix and the control law `v = -λ L⁺ (s − s*)`.)
2. F. Chaumette and S. Hutchinson, "Visual servo control. II. Advanced approaches," *IEEE Robotics & Automation Magazine*, vol. 14, no. 1, pp. 109–118, Mar. 2007. (Choice of interaction matrix: current, desired, mean.)
3. S. Hutchinson, G. D. Hager and P. I. Corke, "A tutorial on visual servo control," *IEEE Transactions on Robotics and Automation*, vol. 12, no. 5, pp. 651–670, Oct. 1996.
4. P. Corke, *Robotics, Vision and Control: Fundamental Algorithms in Python*, 3rd ed. Springer, 2023.

## Kinematics, Jacobians and singularities
5. K. M. Lynch and F. C. Park, *Modern Robotics: Mechanics, Planning, and Control*. Cambridge University Press, 2017. (Twists and the adjoint used for the camera Jacobian.)
6. B. Siciliano, L. Sciavicco, L. Villani and G. Oriolo, *Robotics: Modelling, Planning and Control*. Springer, 2009.
7. J. J. Craig, *Introduction to Robotics: Mechanics and Control*, 4th ed. Pearson, 2018. (Polynomial trajectories.)
8. T. Yoshikawa, "Manipulability of robotic mechanisms," *The International Journal of Robotics Research*, vol. 4, no. 2, pp. 3–9, 1985.
9. Y. Nakamura and H. Hanafusa, "Inverse kinematic solutions with singularity robustness for robot manipulator control," *Journal of Dynamic Systems, Measurement, and Control*, vol. 108, no. 3, pp. 163–171, 1986. (Adaptive damping near singularities.)
10. C. W. Wampler, "Manipulator inverse kinematic solutions based on vector formulations and damped least-squares methods," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 16, no. 1, pp. 93–101, 1986.
11. T. Sugihara, "Solvability-unconcerned inverse kinematics by the Levenberg–Marquardt method," *IEEE Transactions on Robotics*, vol. 27, no. 5, pp. 984–991, 2011.
12. Universal Robots, "Parameters for calculations of kinematics and dynamics" (DH parameters of the UR3e), Universal Robots support site.

## Vision and tracking
13. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press, 2003. (Pinhole camera model.)
14. G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.
15. H. W. Kuhn, "The Hungarian method for the assignment problem," *Naval Research Logistics Quarterly*, vol. 2, no. 1–2, pp. 83–97, 1955.
16. D. F. Crouse, "On implementing 2D rectangular assignment algorithms," *IEEE Transactions on Aerospace and Electronic Systems*, vol. 52, no. 4, pp. 1679–1696, 2016. (Algorithm behind SciPy's `linear_sum_assignment`.)
17. Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*. Wiley, 2001. (Alpha-beta tracking filter.)

## Robotics software
18. P. Corke and J. Haviland, "Not your grandmother's toolbox – the Robotics Toolbox reinvented for Python," in *Proc. IEEE International Conference on Robotics and Automation (ICRA)*, 2021, pp. 11357–11363.
19. Swift (web-based robot simulator), `spatialgeometry` and `spatialmath-python`, by J. Haviland, P. Corke and contributors (open-source projects on GitHub).
20. UTS 41013 Industrial Robotics, *Python Wiki – Simulated Robot Models* and the `ir-support` package (UR3e mesh model used for visualisation only).

## Safety and medical-robotics context
Used as background for the safety design (speed limits, protective stops, restricted zones). The project does **not** claim compliance with any of them.

21. ISO 10218-1:2011 and ISO 10218-2:2011, *Robots and robotic devices – Safety requirements for industrial robots* (Part 1: Robots; Part 2: Robot systems and integration).
22. ISO/TS 15066:2016, *Robots and robotic devices – Collaborative robots*.
23. IEC 80601-2-77:2019, *Medical electrical equipment – Part 2-77: Particular requirements for the basic safety and essential performance of robotically assisted surgical equipment*.
24. R. H. Taylor, A. Menciassi, G. Fichtinger, P. Fiorini and P. Dario, "Medical robotics and computer-integrated surgery," in *Springer Handbook of Robotics*, 2nd ed., B. Siciliano and O. Khatib, Eds. Springer, 2016.
25. G.-Z. Yang *et al.*, "Medical robotics—Regulatory, ethical, and legal considerations for increasing levels of autonomy," *Science Robotics*, vol. 2, no. 4, eaam8638, 2017.

## Software used (versions pinned in `requirements.txt`)
Licences are those published by each project; verify them before redistributing anything.

| Package | Version | Used for |
|---|---|---|
| numpy | 1.26.4 | Numerical computing |
| scipy | 1.17.1 | Rotations, assignment problem |
| pandas | 2.3.3 | Log loading and aggregation |
| matplotlib | 3.10.9 | Evaluation figures |
| opencv-python | 4.11.0.86 | Image rendering, detection, video |
| roboticstoolbox-python | 1.1.1 | Robot model, kinematics |
| spatialmath-python | 1.1.17 | SE(3) transformations |
| swift-sim | 1.1.0 | 3D visualisation |
| spatialgeometry | 1.1.0 | Scene shapes |
| ir-support | 1.4.3 | UR3e mesh model (UTS course package; check its licence) |
| Flask | 3.1.3 | REST API and frontend hosting |
| websockets | 13.1 | Live state streaming |
| pytest | 8.4.2 | Automated tests |