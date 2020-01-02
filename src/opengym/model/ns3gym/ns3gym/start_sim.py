#!/usr/bin/python
import sys
import os
import time
import subprocess


def find_waf_path(cwd):
	wafPath = cwd

	found = False
	myDir = cwd
	while (not found):
		for fname in os.listdir(myDir):
			if fname == "waf":
				found = True
				wafPath = os.path.join(myDir, fname)
				break

		myDir = os.path.dirname(myDir)

	return wafPath


# NOTE unused
def build_ns3_project(debug=True):
	cwd = os.getcwd()
	simScriptName = os.path.basename(cwd)
	wafPath = find_waf_path(cwd)
	baseNs3Dir = os.path.dirname(wafPath)

	os.chdir(baseNs3Dir)

	wafString = wafPath + ' build'

	output = subprocess.DEVNULL
	if debug:
		output = None

	buildRequired = False
	ns3Proc = subprocess.Popen(wafString, shell=True, stdout=subprocess.PIPE, stderr=None, universal_newlines=True)

	lineHistory = []
	for line in ns3Proc.stdout:
		if (True or "Compiling" in line or "Linking" in line) and not buildRequired:
			buildRequired = True
			print("Build ns-3 project if required")
			for l in lineHistory:
				sys.stdout.write(l)
				lineHistory = []

		if buildRequired:
			sys.stdout.write(line)
		else:
			lineHistory.append(line)

	p_status = ns3Proc.wait()
	if buildRequired:
		print("(Re-)Build of ns-3 finished with status: ", p_status)
	os.chdir(cwd)

"""
custom start script without waf
NOTE this function does not invoke waf script for compilation, and just executing the binary of the simulation code
"""
def start_sim_script_without_waf(port=5555, simSeed=0, simArgs={}, debug=False, cwd=None):

	if cwd is None:
		cwd = os.getcwd()
	simScriptName = os.path.basename(cwd)
	wafPath = find_waf_path(cwd)
	baseNs3Dir = os.path.dirname(wafPath)

	binaryPath = os.path.relpath(cwd, baseNs3Dir)
	absBinaryPath = os.path.join(baseNs3Dir, 'build', binaryPath, simScriptName)
	libraryPath = os.path.join(baseNs3Dir, 'build/lib')

	portString = ""
	if port:
		portString = " --openGymPort=" + str(port)
	seedString = ""
	if simSeed:
		seedString = " --simSeed=" + str(simSeed)
	
	argString = ""
	for k, v in simArgs.items():
		argString += " " + str(k) + "=" + str(v)

	# resolve shared library locations
	my_env = os.environ.copy()
	my_env["LD_LIBRARY_PATH"] = libraryPath + ":" + my_env.get("LD_LIBRARY_PATH", "")

	runString = absBinaryPath + portString + seedString + argString

	ns3Proc = None
	if debug:
		ns3Proc = subprocess.Popen(runString, shell=True, stdout=None, stderr=None, env=my_env, cwd=baseNs3Dir)
	else:
		errorOutput = subprocess.DEVNULL
		ns3Proc = subprocess.Popen(runString, shell=True, stdout=subprocess.PIPE, stderr=errorOutput, universal_newlines=True, env=my_env, cwd=baseNs3Dir)

	if debug:
		print("Start command: ", runString)
		print("Started ns3 simulation script, Process Id: ", ns3Proc.pid)

	return ns3Proc


def start_sim_script(port=5555, simSeed=0, simArgs={}, debug=False, cwd=None):
	if cwd is None:
		cwd = os.getcwd()
	simScriptName = os.path.basename(cwd)
	wafPath = find_waf_path(cwd)
	baseNs3Dir = os.path.dirname(wafPath)

	# os.chdir(baseNs3Dir)

	wafString = wafPath + ' --run "' + simScriptName

	if port:
		wafString += ' --openGymPort=' + str(port)

	if simSeed:
		wafString += ' --simSeed=' + str(simSeed)

	for k,v in simArgs.items():
		wafString += " "
		wafString += str(k)
		wafString += "="
		wafString += str(v)

	wafString += '"'

	ns3Proc = None
	if debug:
		ns3Proc = subprocess.Popen(wafString, shell=True, stdout=None, stderr=None, cwd=baseNs3Dir)
	else:
		'''
		users were complaining that when they start example they have to wait 10 min for initialization.
		simply ns3 is being built during this time, so now the output of the build will be put to stdout
		but sometimes build is not required and I would like to avoid unnecessary output on the screen
		it is not easy to get tell before start ./waf whether the build is required or not
		here, I use simple trick, i.e. if output of build contains {"Compiling","Linking"}
		then the build is required and, hence, i put the output to the stdout
		'''
		errorOutput = subprocess.DEVNULL
		ns3Proc = subprocess.Popen(wafString, shell=True, stdout=subprocess.PIPE, stderr=errorOutput, universal_newlines=True, cwd=baseNs3Dir)

		buildRequired = False
		lineHistory = []
		for line in ns3Proc.stdout:
			if ("Compiling" in line or "Linking" in line) and not buildRequired:
				buildRequired = True
				print("Build ns-3 project if required")
				for l in lineHistory:
					sys.stdout.write(l)
					lineHistory = []

			if buildRequired:
				sys.stdout.write(line)
			else:
				lineHistory.append(line)

			if ("Waf: Leaving directory" in line):
				break

	if debug:
		print("Start command: ",wafString)
		print("Started ns3 simulation script, Process Id: ", ns3Proc.pid)

	# go back to my dir
	# os.chdir(cwd)
	return ns3Proc