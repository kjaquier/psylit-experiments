"""
Contains things to load from JVM.
Must be executed only once! Will crash if reloaded in the same kernel.
"""


import pathlib
from jpype import startJVM, shutdownJVM, getDefaultJVMPath, JPackage
import atexit
import logging

from parameters import JVM_PARAMETERS

logger = logging.getLogger('JVM')


def locate_jar(jar_name):
    jar = (next(pathlib.Path().glob('**/bin'), pathlib.Path('bin')) / jar_name).resolve()
    assert jar.exists(), f"Jar '{jar_name}' not found"
    return jar


def start_jvm(classpath, *args):
    logger.info(f"Starting JVM: {' '.join(args)}")
    startJVM(getDefaultJVMPath(), f"-Djava.class.path={classpath}", *args, convertStrings=False)


INFODYN_JAR_LOC = locate_jar('infodynamics.jar')

start_jvm(INFODYN_JAR_LOC, *JVM_PARAMETERS['jvm_args'])


infodynamics_measures_discrete = JPackage("infodynamics.measures.discrete")


@atexit.register
def shutdown():
    logger.info(f"Stopping JVM")
    shutdownJVM()
