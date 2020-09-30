#include "BOC.h"
#include <io.h>
namespace BKHao
{
    int _BOC::search_files_(const std::string& path, const std::string& ext, std::vector<std::string>& filenames)
    {
		intptr_t   hFile = 0;
		struct _finddata_t fileinfo;
		std::string p;
		int num = 0;
		if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		{
			do
			{
				if (fileinfo.attrib & _A_SUBDIR)
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
						num += _BOC::search_files_(p.assign(path).append("\\").append(fileinfo.name), ext, filenames);
				}
				else
				{
					if (strstr(fileinfo.name, ext.c_str()))
					{
						filenames.push_back(p.assign(path).append("\\").append(fileinfo.name));
						++num;
					}
				}
			} while (_findnext(hFile, &fileinfo) == 0);
			_findclose(hFile);
		}
        return num;
    }
}