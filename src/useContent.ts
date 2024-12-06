import {useOperatorExecutor} from "@fiftyone/operators";
import {
  usePanelTitle
} from "@fiftyone/spaces";
import { useEffect } from "react";

type ContentItem = {
  label: string;
  icon: string;
}
export type Content = {
  mainHeaderLabel: string;
  loading: boolean;
  examples: ContentItem[];
  capabilities: ContentItem[];
  iconURL?: string
  iconWidth?: number
}

export default function useContent(): Content {
  const [title, setPanelTitle] = usePanelTitle();
  const fetchContentExecutor = useOperatorExecutor("@voxel51/voxelgpt/fetch_content");
  useEffect(() => {
    fetchContentExecutor.execute();
  }, []);
  useEffect(() => {
    const title = fetchContentExecutor.result?.main_header_label
    if (title) setPanelTitle(title);
  }, [fetchContentExecutor.result?.main_header_label]);
  console.log(fetchContentExecutor)

  return {
    loading: fetchContentExecutor.loading,
    mainHeaderLabel: fetchContentExecutor?.result?.main_header_label,
    examples: fetchContentExecutor?.result?.examples,
    capabilities: fetchContentExecutor?.result?.capabilities,
    iconURL: fetchContentExecutor?.result?.icon_url,
    iconWidth: fetchContentExecutor?.result?.icon_width
  };
}
