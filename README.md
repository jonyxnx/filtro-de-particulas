# Particle Filter for Trajectory Estimation

## Project Overview

This repository presents a comprehensive study on the implementation and analysis of a particle filter for trajectory estimation based on the Hidden Markov Model (HMM). The filter's performance is evaluated in both controlled environments and randomly generated mazes, showcasing its adaptability in dynamic scenarios.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Objectives](#2-objectives)
- [3. Theoretical Framework](#3-theoretical-framework)
- [4. Development](#4-development)
- [5. Results](#5-results)
- [6. Conclusion](#6-conclusion)
- [7. Appendices](#7-appendices)
- [8. References](#8-references)

## 1. Introduction

In fields like robotics, computational perception, and signal recovery, robust localization algorithms play a crucial role. The particle filter, explored in this project, emerges as a versatile solution applicable to various domains, ranging from autonomous vehicles to object tracking systems.

## 2. Objectives

The primary objectives of the project include:

1. Develop a robust particle filter for estimating real trajectories in a confined space.
2. Increase the environmental complexity and noise affecting the position estimates.
3. Apply the particle filter in complex situations and compare its performance.
4. Analyze the effectiveness and efficiency of the filter under different scenarios and conditions.

## 3. Theoretical Framework

### 3.1 Hidden Markov Model (HMM)

The HMM is a statistical model describing a stochastic process with unobservable states evolving over time. It consists of hidden states and observable processes, forming the basis for the particle filter.

### 3.2 Particle Filter

The particle filter, within the HMM framework, estimates the hidden state (trajectory) based on observations. It involves stages like particle initialization, state prediction, state update, weight normalization, resampling, and final state estimation.

### 3.3 Wilson Algorithm

The Wilson algorithm generates random mazes, enhancing the experimentation environment for a more realistic analysis of the particle filter.

## 4. Development

The project's implementation involves:

- Simulation of Brownian motion as the primary stochastic process for trajectory modeling.
- Application of the particle filter to estimate trajectories in controlled and maze environments.
- Visualization of results using Jupyter notebooks.

## 5. Results

The particle filter demonstrates effectiveness in trajectory estimation, as showcased in various scenarios, from controlled environments to complex mazes. The comparison of real trajectories with estimated ones illustrates the filter's adaptability and accuracy.

## 6. Conclusion

The study contributes to the advancement of localization research by providing a robust and accessible implementation of the particle filter. The adaptability of the filter in dynamic and uncertain environments positions it as a valuable tool for diverse applications.

## 7. Appendices

### 7.1 Wilson Algorithm

The Wilson algorithm efficiently generates random mazes based on random walks, contributing to the creation of complex and unique environments for experimentation.

## 8. References

• Doucet, A.,et al. (sf.) An Introduction To Sequential Monte Carlo Methods, Oxford University.
• Wills, Adrian G.; Schön, Thomas B. (3 May 2023). “Sequential Monte Carlo: A Unified Review”. Annual Review of Control, Robotics, and Autonomous Systems. 6: 159–182. doi:10.1146/annurev-control-042920-015119. ISSN 2573-5144. S2CID 255638127.
• Del Moral, Pierre (1996). “Non Linear Filtering: Interacting Particle Solution”. Markov Processes and Related
• Doucet, A.; Johansen, A.M. (December 2008). “A tutorial on particle filtering and smoothing: fifteen years later” . Technical Report

