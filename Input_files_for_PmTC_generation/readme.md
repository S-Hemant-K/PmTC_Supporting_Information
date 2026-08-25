Commands to generate PmTC models on ICM-Pro:

`icm64 protacModel.icm <stationary_protein.ob> <mobile_protein.ob> <session_filename.icb> protac=<protac.sdf> scsRad=12.0 effort=1`

- `<stationary_protein.ob>` replace this with your protein `.ob` file that is to be fixed in space during linker torsional sampling

- `<mobile_protein.ob>` replace this with your protein `.ob` file that is to be allowed to move with the linker during its torsional sampling

- `<session_filename.icb>` replace this with your choice of ICM session file name. Convinient to use project prefix followed by iteration number. Eg., `5t35_mdl_1.icb`. Such a convention will be convinient to load into ICM Browser

- `<protac.sdf>` replace this with your SDF file of your PROTAC

After at least 5 iterations using the above code, initiate an ICM session in the directory where the models were generated and run the following:

`processPROTACsims "<session_filename_prefix>*.icb"`

Eg., `processPROTACsims "5t35_mdl*.icb"`

**Credits: MolSoft**
